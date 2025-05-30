# from difflib import SequenceMatcher

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import csv 
import time
import pandas as pd
import json
import argparse

import re
from typing import Set

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
PUNCTUATIONS = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate human benchmarks.")
    parser.add_argument("--pred_path", default=r'./eval_human/internvl-38b-cot.json', help="The path to file containing prediction.")
    parser.add_argument("--fixed_path", default=r'./eval_human/internvl-38b-cot_fixed.json', help="The path to file containing prediction.")
    parser.add_argument("--save_path", default=r'./eval_human/cot-internvl-38b-cot.csv', help="The path to file containing prediction.")
    
    
    args = parser.parse_args()
    return args

def process_answer(answer: str) -> str:
    if isinstance(answer, list):
        answer = answer[0]
    answer = answer.lower()
    answer = COMMA_STRIP.sub('', answer)
    answer = PERIOD_STRIP.sub('', answer)
    for p in PUNCTUATIONS:
        answer = answer.replace(p, '')
    words = answer.split()

    processed_words = []
    for word in words:
        if word in CONTRACTIONS:
            processed_words.append(CONTRACTIONS[word])
        else:
            processed_words.append(word)
    answer = ' '.join(processed_words)

    words = answer.split()
    processed_words = []
    for word in words:
        if word in NUMBER_MAP:
            processed_words.append(NUMBER_MAP[word])
        else:
            processed_words.append(word)
    answer = ' '.join(processed_words)

    words = answer.split()
    if words and words[0] in ARTICLES:
        words = words[1:]
    answer = ' '.join(words)
    answer = ' '.join(answer.split()).strip()
    return answer


def normalize_step(step):
    step = step.strip(" \n。，,.!?")
    step = step.lower()
    return step


def split_causal_chain(text):
    if not isinstance(text, list):
        return [process_answer(x) for x in text.split("→") if x.strip()]
    else:
        return [process_answer(x) for x in text[0].split("→") if x.strip()]


encoder = SentenceTransformer("all-MiniLM-L6-v2")

def is_semantic_match(pred_step, gt_step, threshold=0.6):
    emb_pred = encoder.encode([pred_step])
    emb_gt = encoder.encode([gt_step])
    sim = cosine_similarity(emb_pred, emb_gt)[0][0]
    return sim >= threshold

def is_semantic_match_score(pred_step, gt_step, threshold=0.6):
    emb_pred = encoder.encode([pred_step])
    emb_gt = encoder.encode([gt_step])
    sim = cosine_similarity(emb_pred, emb_gt)[0][0]
    return sim

def fuzzy_precision_recall_f1(pred_steps, gt_steps, threshold=0.6):
    candidate_matches = []
    for i, p_step in enumerate(pred_steps):
        for j, g_step in enumerate(gt_steps):
            similarity = is_semantic_match_score(p_step, g_step)
            if similarity >= threshold:
                candidate_matches.append((similarity, i, j))
    candidate_matches.sort(reverse=True, key=lambda x: x[0])
    
    matched_pred = set()
    matched_gt = set()
    
    for sim, i, j in candidate_matches:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
    
    precision = len(matched_pred)/len(pred_steps) if pred_steps else 0
    recall = len(matched_gt)/len(gt_steps) if gt_steps else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    return precision, recall, f1

def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if is_semantic_match(a[i], b[j]):
                dp[i+1][j+1] = dp[i][j]+1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[-1][-1]

# def lcs_order_score(pred_steps, gt_steps):
#     return lcs_length(pred_steps, gt_steps) / len(gt_steps) if gt_steps else 0


def lcs_order_score(pred_steps, gt_steps, threshold=0.5):
    matched_pairs = []
    for i, p_step in enumerate(pred_steps):
        for j, g_step in enumerate(gt_steps):
            if is_semantic_match(p_step, g_step, threshold):
                matched_pairs.append((i, j))
    
    if not matched_pairs:
        return 0.0
    
    matched_pairs.sort(key=lambda x: x[0])
    gt_sequence = [j for _, j in matched_pairs]
    
    n = len(gt_sequence)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if gt_sequence[j] < gt_sequence[i] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
    
    max_lcs = max(dp) if n > 0 else 0
    return max_lcs / len(gt_steps) if gt_steps else 0

#############################MLLMs#############################
model_name = "/Qwen2.5-72B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#############################MLLMs#############################


def fix_json_lines(input_path: str, output_path: str):
    fixed_list = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                fixed_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("error")
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(fixed_list, fout, ensure_ascii=False, indent=2)
    print(f"save at {output_path}")
        


def main():
    args = parse_args()
    
    fix_json_lines(args.pred_path, args.fixed_path)

    # === Load result file ===
    with open(args.fixed_path, "r", encoding="utf-8") as f:
        res = json.load(f)

        # === Evaluate each sample ===
        for item in res:
            pred = split_causal_chain(item["pred"]) 
            # gt_answers = list({split_causal_chain(ans) for ans in item["answer"]})
            gt_answers = split_causal_chain(item["answer"])

            prompt = (
                "Please determine whether the causal explanation generated by the model is semantically consistent with the reference causal chain. "
                "Model's causal explanation: {} "
                "Reference causal chain: {} "
                "Please provide a consistency score (0-5) directly, and do not include any explanation."
            ).format(pred, gt_answers)
            

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ] 

            gpt_score = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
            
            precision, recall, f1 = fuzzy_precision_recall_f1(pred, gt_answers)
            
            
            order_score = lcs_order_score(pred, gt_answers)
        
            f1 = round(f1, 2) 
            order_score = round(order_score, 2) 
            gpt_score_float = round(float(gpt_score.strip()), 2)
            
            final_score = round(0.5*f1 + 0.3*order_score + 0.5*(gpt_score_float/5), 2)

            with open(args.save_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([f1, order_score, round(gpt_score_float/5,2), final_score])
                
            
if __name__ == '__main__':
    main()