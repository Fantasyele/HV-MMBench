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
            if len(item['choices']) in {2, 3, 4}
        ]
        # self.processor = processor
    
    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['videoid']
        question = line['generatedquestion']

        instructs = []
        ops = []
        task_types = []
        answer_idxs = []
        
        options = line['choices']
        task_type = line['attributecategory']
        if len(options) == 4:
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        elif len(options) == 3:
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        elif len(options) == 2:
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        else:
            idx += 1
            return self.__getitem__(idx)
            
        answer_idx = line['answeridx']

        instructs.append(instruct)
        ops.append(options)
        task_types.append(task_type)
        answer_idxs.append(answer_idx)
                
        return {
            # 'video': video_tensor, 
            'video_id': video_name,
            'instructs': instructs,
            'options': ops,
            'task_types': task_types,
            'answer_idxs': answer_idxs,
            
        }


def collate_fn(batch):
    # vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    ins = [x['instructs'] for x in batch]
    task_types = [x['task_types'] for x in batch]
    ops = [x['options'] for x in batch]
    answer_idxs =  [x['answer_idxs'] for x in batch]
    
    # vid = torch.stack(vid, dim=0)
    return v_id, ins, task_types, ops, answer_idxs

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
    questions = get_chunk(flattened_data, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"

    dataset = HVMMBDataset(questions)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (video_id, instructs, task_types, options, answer_idxs) in enumerate(tqdm(dataloader)):
        # reduce batch dimension
        # video_tensor = video_tensor[0]
        video_id = video_id[0]


        if os.path.exists(os.path.join(args.video_folder, f"{video_id}")):
            video_path = os.path.join(args.video_folder, f"{video_id}")
        else:
            video_path = video_id

        instructs = instructs[0]
        options = options[0]
        task_type = task_types[0]
        gt_answer_idx = answer_idxs[0]
        # import pdb;pdb.set_trace()
        qas = []
        for idx, instruct in enumerate(instructs):
            letters = ['(A)', '(B)', '(C)', '(D)']
            _options = options[idx]
            
            prompt = instruct
            
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
            
            output = output[0].replace('answer', '')
            output = output.replace('Answer', '')
            pred_answer = re.findall('\(*[A-D]\)*', output)
            try:
                assert len(pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(video_id, instruct, output)
                pred_answer = pred_answer[0].strip()
                # if not pred_answer.startswith('('):
                pred_answer = pred_answer.strip('()')
                pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)
            except:
                traceback.print_exc()
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2
            results = {'vid': video_id, "task_type": task_type[0], "gt": gt_answer_idx[0], 'answer_id': pred_idx, 'answer': _options[pred_idx]}
            print(results)

            qas.append({'vid': video_id, "task_type": task_type[0], "gt": gt_answer_idx[0], 'answer_id': pred_idx, 'answer': _options[pred_idx]})

        ans_file.write('{},\n'.format(json.dumps(qas, ensure_ascii=False)))
            # with open(ans_file, 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(qas, ensure_ascii=False) + '\n')

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
