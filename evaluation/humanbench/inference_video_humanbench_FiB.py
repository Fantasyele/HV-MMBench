import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class HVMMBDataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def __init__(self, data_list):
        # self.data_list = data_list
        self.data_list = [
            item for item in data_list 
            if len(item['generatedquestion']) > 3
        ]
    
    def __len__(self):
        return len(self.data_list) # 所有QA的item
    
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['videoid']
        question = line['generatedquestion']
        task_type = line['attributecategory']
        answer = line['correctanswer']

        return {
            # 'video':       video_tensor,
            'video_name':  video_name,
            'question':    question,
            'answer':      answer,
            'task_type': task_type
        }

def collate_fn(batch):
    # vid  = [x['video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus  = [x['question'] for x in batch]
    ans  = [x['answer'] for x in batch]
    task_type  = [x['task_type'] for x in batch]
    
    return v_id, qus, ans, task_type



def run_inference(args):

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # default processor
    processor = AutoProcessor.from_pretrained(args.model_path)


    questions = json.load(open(args.question_file, "r"))
    flattened_data = [question for video in questions for question in video]
    # questions = list(flattened_data.values())
    gt_questions = get_chunk(flattened_data, args.num_chunks, args.chunk_idx)
    
    answer_file = os.path.join(args.answer_file)
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = HVMMBDataset(gt_questions)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Iterate over each sample in the ground truth file
    for idx, (video_names, questions, answers, task_types) in enumerate(tqdm(dataloader)):


        video_name   = video_names[0]
        # import pdb;pdb.set_trace()
        # if video_names in exist_ids:
        #     continue
        
        question     = questions[0] + 'Answer in short:'
        answer       = answers[0]
        task_type = task_types[0]
    

        if os.path.exists(os.path.join(args.video_folder, f"{video_name}")):
            video_path = os.path.join(args.video_folder, f"{video_name}")
        else:
            video_path = video_name

        prompt = question
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        # "max_pixels": 480 * 270, 
                        # 'resized_height': 270,
                        # 'resized_width': 480,
                        "total_pixels": 20480 * 28 * 28, 
                        "min_pixels": 16 * 28 * 28,
                        "fps": 0.2,
                    },
                    {"type": "text", "text": prompt
                        
                    },
                ], 
            } 
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True) # 
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs, 
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output)
            
        sample_set = {'video_name': video_name, 'task_type': task_type, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
