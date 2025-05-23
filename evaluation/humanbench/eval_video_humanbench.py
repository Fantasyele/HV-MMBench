import json
import argparse
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate human benchmarks.")
    parser.add_argument("--pred_path", default=r'eval_results_32B/pexels/pd/results.json', help="The path to file containing prediction.")
    parser.add_argument("--fixed_path", default=r'eval_results_32B/pexels/pd/results_fixed.json', help="The path to file containing prediction.")
    parser.add_argument("--save_csv_path", default=r'eval_results_32B/statistic/pd/all_results.csv', help="The path to file containing prediction.")
    
    
    args = parser.parse_args()
    return args


tasks = ['Basic Attribute Recognition', 'Face Recognition', 'Action Recognition', 'Relationship Inference', 'Intention Inference', 'Causal Reasoning']

def main():
    args = parse_args()
    # res = [eval(x.strip()) for x in open(args.pred_path, 'r').readlines()]

    with open(args.pred_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_data = []
    for line in lines:
        line = line.strip().rstrip(',') 
        if not line:
            continue
        try:
            obj = json.loads(line)
            fixed_data.append(obj)
        except json.JSONDecodeError as e:
            print('error')

    with open(args.fixed_path, 'w', encoding='utf-8') as f_out:
        json.dump(fixed_data, f_out, ensure_ascii=False, indent=2)

    
    # res = [eval(x.strip()) for x in open(args.fixed_path, 'r').readlines()]

    with open(args.fixed_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    
    task_types = tasks
    task_acc = {x: [] for x in task_types}
    
    acc = []
    for i, x in enumerate(res):
        value = 1
        
        if x[0]['answer_id'] != x[0]['gt']:
            value = 0
        acc.append(value)
        if x[0]['task_type'] == 'Emotional Expression':
            continue
        task_acc[x[0]['task_type']].append(value)
    acc = sum(acc) * 100 / len(acc)
    
    # task_acc = {x: sum(task_acc[x]) * 100 / len(task_acc[x]) for x in task_acc}

    task_acc = {
        x: (sum(task_acc[x]) * 100 / len(task_acc[x])) if len(task_acc[x]) > 0 else None
        for x in task_acc
    }

    print(f"{args.pred_path}:", acc)
    task_names = list(tasks)
    
    table_data = []
    for i in range(len(task_names) // 6):
        row_task_names = task_names[i * 6: (i + 1) * 6]
        # row_task_acc = ['{:.2f}'.format(task_acc[x]) for x in row_task_names if task_acc[x] is not None else None]
        row_task_acc = [
            '{:.2f}'.format(task_acc[x]) if isinstance(task_acc[x], (int, float)) else task_acc[x]
            for x in row_task_names
        ]
        table_data.append(row_task_names)
        table_data.append(row_task_acc)
    print(tabulate(table_data, floatfmt=".1f"), '\n')
    
    
    import csv
    import os
    csv_path = args.save_csv_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(table_data)


if __name__ == '__main__':
    main()
