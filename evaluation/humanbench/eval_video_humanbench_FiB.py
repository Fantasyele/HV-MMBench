# Copyright (c) Facebook, Inc. and its affiliates.
import re
from tqdm import tqdm

# class EvalAIAnswerProcessor:
#     """
#     Processes an answer similar to Eval AI
#         copied from
#         https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
#     """

#     CONTRACTIONS = {
#         "aint": "ain't",
#         "arent": "aren't",
#         "cant": "can't",
#         "couldve": "could've",
#         "couldnt": "couldn't",
#         "couldn'tve": "couldn't've",
#         "couldnt've": "couldn't've",
#         "didnt": "didn't",
#         "doesnt": "doesn't",
#         "dont": "don't",
#         "hadnt": "hadn't",
#         "hadnt've": "hadn't've",
#         "hadn'tve": "hadn't've",
#         "hasnt": "hasn't",
#         "havent": "haven't",
#         "hed": "he'd",
#         "hed've": "he'd've",
#         "he'dve": "he'd've",
#         "hes": "he's",
#         "howd": "how'd",
#         "howll": "how'll",
#         "hows": "how's",
#         "Id've": "I'd've",
#         "I'dve": "I'd've",
#         "Im": "I'm",
#         "Ive": "I've",
#         "isnt": "isn't",
#         "itd": "it'd",
#         "itd've": "it'd've",
#         "it'dve": "it'd've",
#         "itll": "it'll",
#         "let's": "let's",
#         "maam": "ma'am",
#         "mightnt": "mightn't",
#         "mightnt've": "mightn't've",
#         "mightn'tve": "mightn't've",
#         "mightve": "might've",
#         "mustnt": "mustn't",
#         "mustve": "must've",
#         "neednt": "needn't",
#         "notve": "not've",
#         "oclock": "o'clock",
#         "oughtnt": "oughtn't",
#         "ow's'at": "'ow's'at",
#         "'ows'at": "'ow's'at",
#         "'ow'sat": "'ow's'at",
#         "shant": "shan't",
#         "shed've": "she'd've",
#         "she'dve": "she'd've",
#         "she's": "she's",
#         "shouldve": "should've",
#         "shouldnt": "shouldn't",
#         "shouldnt've": "shouldn't've",
#         "shouldn'tve": "shouldn't've",
#         "somebody'd": "somebodyd",
#         "somebodyd've": "somebody'd've",
#         "somebody'dve": "somebody'd've",
#         "somebodyll": "somebody'll",
#         "somebodys": "somebody's",
#         "someoned": "someone'd",
#         "someoned've": "someone'd've",
#         "someone'dve": "someone'd've",
#         "someonell": "someone'll",
#         "someones": "someone's",
#         "somethingd": "something'd",
#         "somethingd've": "something'd've",
#         "something'dve": "something'd've",
#         "somethingll": "something'll",
#         "thats": "that's",
#         "thered": "there'd",
#         "thered've": "there'd've",
#         "there'dve": "there'd've",
#         "therere": "there're",
#         "theres": "there's",
#         "theyd": "they'd",
#         "theyd've": "they'd've",
#         "they'dve": "they'd've",
#         "theyll": "they'll",
#         "theyre": "they're",
#         "theyve": "they've",
#         "twas": "'twas",
#         "wasnt": "wasn't",
#         "wed've": "we'd've",
#         "we'dve": "we'd've",
#         "weve": "we've",
#         "werent": "weren't",
#         "whatll": "what'll",
#         "whatre": "what're",
#         "whats": "what's",
#         "whatve": "what've",
#         "whens": "when's",
#         "whered": "where'd",
#         "wheres": "where's",
#         "whereve": "where've",
#         "whod": "who'd",
#         "whod've": "who'd've",
#         "who'dve": "who'd've",
#         "wholl": "who'll",
#         "whos": "who's",
#         "whove": "who've",
#         "whyll": "why'll",
#         "whyre": "why're",
#         "whys": "why's",
#         "wont": "won't",
#         "wouldve": "would've",
#         "wouldnt": "wouldn't",
#         "wouldnt've": "wouldn't've",
#         "wouldn'tve": "wouldn't've",
#         "yall": "y'all",
#         "yall'll": "y'all'll",
#         "y'allll": "y'all'll",
#         "yall'd've": "y'all'd've",
#         "y'alld've": "y'all'd've",
#         "y'all'dve": "y'all'd've",
#         "youd": "you'd",
#         "youd've": "you'd've",
#         "you'dve": "you'd've",
#         "youll": "you'll",
#         "youre": "you're",
#         "youve": "you've",
#     }

#     NUMBER_MAP = {
#         "none": "0",
#         "zero": "0",
#         "one": "1",
#         "two": "2",
#         "three": "3",
#         "four": "4",
#         "five": "5",
#         "six": "6",
#         "seven": "7",
#         "eight": "8",
#         "nine": "9",
#         "ten": "10",
#     }
#     ARTICLES = ["a", "an", "the"]
#     PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
#     COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
#     PUNCTUATIONS = [
#         ";",
#         r"/",
#         "[",
#         "]",
#         '"',
#         "{",
#         "}",
#         "(",
#         ")",
#         "=",
#         "+",
#         "\\",
#         "_",
#         "-",
#         ">",
#         "<",
#         "@",
#         "`",
#         ",",
#         "?",
#         "!",
#     ]

#     def __init__(self, *args, **kwargs):
#         pass

#     def word_tokenize(self, word):
#         word = word.lower()
#         word = word.replace(",", "").replace("?", "").replace("'s", " 's")
#         return word.strip()

#     def process_punctuation(self, in_text):
#         out_text = in_text
#         for p in self.PUNCTUATIONS:
#             if (p + " " in in_text or " " + p in in_text) or (
#                 re.search(self.COMMA_STRIP, in_text) is not None
#             ):
#                 out_text = out_text.replace(p, "")
#             else:
#                 out_text = out_text.replace(p, " ")
#         out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
#         return out_text

#     def process_digit_article(self, in_text):
#         out_text = []
#         temp_text = in_text.lower().split()
#         for word in temp_text:
#             word = self.NUMBER_MAP.setdefault(word, word)
#             if word not in self.ARTICLES:
#                 out_text.append(word)
#             else:
#                 pass
#         for word_id, word in enumerate(out_text):
#             if word in self.CONTRACTIONS:
#                 out_text[word_id] = self.CONTRACTIONS[word]
#         out_text = " ".join(out_text)
#         return out_text

#     def __call__(self, item):
#         item = self.word_tokenize(item)
#         item = item.replace("\n", " ").replace("\t", " ").strip()
#         item = self.process_punctuation(item)
#         item = self.process_digit_article(item)
#         return item


# class TextVQAAccuracyEvaluator:
#     def __init__(self):
#         self.answer_processor = EvalAIAnswerProcessor()

#     def _compute_answer_scores(self, raw_answers):
#         """
#         compute the accuracy (soft score) of human answers
#         """
#         answers = [self.answer_processor(a) for a in raw_answers]
#         assert len(answers) == 10
#         gt_answers = list(enumerate(answers))
#         unique_answers = set(answers)
#         unique_answer_scores = {}

#         for unique_answer in unique_answers:
#             accs = []
#             for gt_answer in gt_answers:
#                 other_answers = [item for item in gt_answers if item != gt_answer]
#                 matching_answers = [
#                     item for item in other_answers if item[1] == unique_answer
#                 ]
#                 acc = min(1, float(len(matching_answers)) / 3)
#                 accs.append(acc)
#             unique_answer_scores[unique_answer] = sum(accs) / len(accs)

#         return unique_answer_scores
    

import re
from typing import Set

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



from typing import List, Set
import argparse
import json
from collections import defaultdict
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

def normalize(text):
    if isinstance(text, List):
        return text[0].strip().lower()
    else:
        return text.strip().lower()
        

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate human benchmarks.")
    parser.add_argument("--pred_path", default=r'./eval_results_qwen2/youtube/tk/results.json', help="The path to file containing prediction.")
    parser.add_argument("--fixed_path", default=r'./eval_results_qwen2/youtube/tk/results_fixed.json', help="The path to file containing prediction.")
    
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
    # Replace contractions
    processed_words = []
    for word in words:
        if word in CONTRACTIONS:
            processed_words.append(CONTRACTIONS[word])
        else:
            processed_words.append(word)
    answer = ' '.join(processed_words)
    # Replace number words
    words = answer.split()
    processed_words = []
    for word in words:
        if word in NUMBER_MAP:
            processed_words.append(NUMBER_MAP[word])
        else:
            processed_words.append(word)
    answer = ' '.join(processed_words)
    # Remove articles
    words = answer.split()
    if words and words[0] in ARTICLES:
        words = words[1:]
    answer = ' '.join(words)
    # Normalize whitespace
    answer = ' '.join(answer.split()).strip()
    return answer


def precision_at_k(predicted_list: List[str], ground_truth_set: Set[str], k: int) -> float:
    if not predicted_list or not ground_truth_set:
        return 0.0
    processed_truths = {process_answer(truth) for truth in ground_truth_set}
    top_k_preds = predicted_list[:k]
    processed_preds = [process_answer(pred) for pred in top_k_preds]
    correct = [pred for pred in processed_preds if pred in processed_truths]
    return len(correct) / k

def recall_at_k(predicted_list: List[str], ground_truth_set: Set[str], k: int) -> float:
    if not predicted_list or not ground_truth_set:
        return 0.0
    processed_truths = {process_answer(truth) for truth in ground_truth_set}
    top_k_preds = predicted_list[:k]
    processed_preds = [process_answer(pred) for pred in top_k_preds]
    correct = [pred for pred in processed_preds if pred in processed_truths]
    return len(correct) / len(processed_truths) if processed_truths else 0.0

def f1_at_k(predicted_list: List[str], ground_truth_set: Set[str], k: int) -> float:
    p = precision_at_k(predicted_list, ground_truth_set, k)
    r = recall_at_k(predicted_list, ground_truth_set, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def hit_at_k(predicted_list: List[str], ground_truth_set: Set[str], k: int) -> int:
    if not predicted_list or not ground_truth_set:
        return 0
    top_k_preds = predicted_list[:k]
    return int(any(pred in ground_truth_set for pred in top_k_preds))



def precision_recall_f1(pred: str, answers: list):
    pred_set = {normalize(pred)}
    gold_set = set(map(normalize, answers))
    
    inter = pred_set & gold_set
    precision = len(inter) / len(pred_set) if pred_set else 0
    recall = len(inter) / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    em = int(normalize(pred) in gold_set)
    
    return em, precision, recall, f1

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
                print('error')
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(fixed_list, fout, ensure_ascii=False, indent=2)
        

def main():
    args = parse_args()
    
    fix_json_lines(args.pred_path, args.fixed_path)
        

    # === Load result file ===
    with open(args.fixed_path, "r", encoding="utf-8") as f:
        res = json.load(f)

        # === Collect task types ===
        # task_types = list({x["task_type"] for x in res})

        # === Init accumulators ===
        # task_metrics = {t: {"em": [], "precision": [], "recall": [], "f1": [], "gpt_score": []} for t in task_types}
        # all_metrics = {"em": [], "precision": [], "recall": [], "f1": [], "gpt_score": []}

        task_types = []
        # === Evaluate each sample ===
        for item in res:

            task = item["task_type"]
            
            if 'relationship' in task.lower():
                task = 'Relationship Inference'
            if 'behavior' in task.lower():
                task = 'Intention Inference'
            if 'action' in task.lower():
                task = 'Action Recognition'
            
            if task not in task_types:
                task_types.append(task)
                
        task_metrics = {t: {"em": [], "precision": [], "recall": [], "f1": []} for t in task_types}
        all_metrics = {"em": [], "precision": [], "recall": [], "f1": []}
        
            
    # === Load result file ===
    with open(args.fixed_path, "r", encoding="utf-8") as f:
        res = json.load(f)

        # === Collect task types ===
        # task_types = list({x["task_type"] for x in res})

        # === Init accumulators ===
        # task_metrics = {t: {"em": [], "precision": [], "recall": [], "f1": [], "gpt_score": []} for t in task_types}
        # all_metrics = {"em": [], "precision": [], "recall": [], "f1": [], "gpt_score": []}

        task_metrics = {t: {"em": [], "precision": [], "recall": [], "f1": []} for t in task_types}
        all_metrics = {"em": [], "precision": [], "recall": [], "f1": []}
        
        # === Evaluate each sample ===
        for item in res:

            task = item["task_type"]
            
            if 'relationship' in task.lower():
                task = 'Relationship Inference'
            if 'behavior' in task.lower():
                task = 'Intention Inference'
            if 'action' in task.lower():
                task = 'Action Recognition'
            
            
            gt_answers = list({normalize(ans) for ans in item["answer"]})
            pred = normalize(item["pred"])
            k = 1
            

            em = exact_match_top1(pred, gt_answers)
            p = precision_at_k([pred], gt_answers, k)
            r = recall_at_k([pred], gt_answers, k)
            f1 = f1_at_k([pred], gt_answers, k)
        
            task_metrics[task]["em"].append(em)
            task_metrics[task]["precision"].append(p)
            task_metrics[task]["recall"].append(r)
            task_metrics[task]["f1"].append(f1)

            all_metrics["em"].append(em)
            all_metrics["precision"].append(p)
            all_metrics["recall"].append(r)
            all_metrics["f1"].append(f1)

        # === Compute per-task and average scores ===
        def mean(lst):
            return sum(lst) * 100 / len(lst) if lst else 0.0

        task_avg = {}
        for t in task_types:
            task_avg[t] = {
                "EM": mean(task_metrics[t]["em"]),
                "Precision": mean(task_metrics[t]["precision"]),
                "Recall": mean(task_metrics[t]["recall"]),
                "F1": mean(task_metrics[t]["f1"]),
            }

        overall_avg = {
            "EM": mean(all_metrics["em"]),
            "Precision": mean(all_metrics["precision"]),
            "Recall": mean(all_metrics["recall"]),
            "F1": mean(all_metrics["f1"]),
        }

        # === Display results ===
        print(f"Overall EM / P / R / F1: {overall_avg}")

        # Pretty table output by 6-task rows
        task_names = sorted(task_types)
        table_data = []

        if len(task_names) == 5:
            for i in range(len(task_names) // 5):
                row_task_names = task_names[i * 5: (i + 1) * 5]
                row_metrics = [
                    # ["EM"] + [f"{task_avg[t]['EM']:.1f}" for t in row_task_names],
                    ["Precision"] + [f"{task_avg[t]['Precision']:.1f}" for t in row_task_names],
                    ["Recall"] + [f"{task_avg[t]['Recall']:.1f}" for t in row_task_names],
                    ["F1"] + [f"{task_avg[t]['F1']:.1f}" for t in row_task_names],
                ]
                table_data.extend(row_metrics)
                table_data.append([])  # Empty row for spacing

            print(tabulate(table_data, headers=["Metric"] + row_task_names, tablefmt="grid"))
            
        elif len(task_names)== 6:

            for i in range(len(task_names) // 6):
                row_task_names = task_names[i * 6: (i + 1) * 6]
                row_metrics = [
                    # ["EM"] + [f"{task_avg[t]['EM']:.1f}" for t in row_task_names],
                    ["Precision"] + [f"{task_avg[t]['Precision']:.1f}" for t in row_task_names],
                    ["Recall"] + [f"{task_avg[t]['Recall']:.1f}" for t in row_task_names],
                    ["F1"] + [f"{task_avg[t]['F1']:.1f}" for t in row_task_names],
                ]
                table_data.extend(row_metrics)
                table_data.append([])  # Empty row for spacing

            print(tabulate(table_data, headers=["Metric"] + row_task_names, tablefmt="grid"))


        elif len(task_names)== 4:

            for i in range(len(task_names) // 4):
                row_task_names = task_names[i * 4: (i + 1) * 4]
                row_metrics = [
                    # ["EM"] + [f"{task_avg[t]['EM']:.1f}" for t in row_task_names],
                    ["Precision"] + [f"{task_avg[t]['Precision']:.1f}" for t in row_task_names],
                    ["Recall"] + [f"{task_avg[t]['Recall']:.1f}" for t in row_task_names],
                    ["F1"] + [f"{task_avg[t]['F1']:.1f}" for t in row_task_names],
                ]
                table_data.extend(row_metrics)
                table_data.append([])  # Empty row for spacing

            print(tabulate(table_data, headers=["Metric"] + row_task_names, tablefmt="grid"))

    
if __name__ == '__main__':


    main()