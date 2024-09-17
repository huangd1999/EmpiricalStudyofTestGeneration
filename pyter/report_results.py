
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


def calculate_coverage_summary(coverage_results):
    total_stmts, total_miss,total_covered = 0, 0,0
    for key in coverage_results["canonical_solution_test_case"]:
        if coverage_results["canonical_solution_test_case"][key]["pass"]:
            total_stmts+= coverage_results["canonical_solution_test_case"][key]["stmts"]
            total_miss+= coverage_results["canonical_solution_test_case"][key]["miss"]
        else:
            total_stmts+= coverage_results["canonical_solution_test_case"][key]["stmts"]
            total_miss+= coverage_results["canonical_solution_test_case"][key]["stmts"]
    print(total_stmts,total_covered)
    if total_stmts !=0:
        coverage = (total_stmts-total_miss)/total_stmts*100
    else:
        coverage = 0

    return coverage

def test_report(dataset):

    correct = 0
    total = 0
    for i in range(len(dataset)):
        correct_test = dataset[i]["correct_tests"]
        total_cases = dataset[i]["correct_tests"] + "\n" + dataset[i]["incorrect_tests"]
        correct += len([line for line in correct_test.split("\n") if "assert" in line])
        total += len([line for line in total_cases.split("\n") if "assert" in line])
    
    if total == 0:
        return 0
    correctness = round(correct/total*100,2)
    return correctness



model_lists = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]



all_correct = {}
new_model_lists = []
all_correct = {}
for model in model_lists:
    if "/" in model:
        model = model.split("/")[-1]
    new_model_lists.append(model)

    all_correct[model] = {}

    # for prompt in ["correct","incorrect"]:
    results = [0,0,0,0,0,0,0,0,0,0,0,0]
    path = f"./source_codes/rq11_21_pyter_{model}_correct.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[0] = dataset["correctness"]
    results[4] = calculate_coverage_summary(dataset["coverage_results"])

    path = f"./source_codes/rq11_21_pyter_{model}_incorrect.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[1] = dataset["correctness"]
    results[5] = calculate_coverage_summary(dataset["coverage_results"])

    path = f"./source_codes/rq12_22_pyter_{model}_correct.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[2] = dataset["correctness"]
    results[6] = calculate_coverage_summary(dataset["coverage_results"])

    path = f"./source_codes/rq12_22_pyter_{model}_incorrect.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[3] = dataset["correctness"]
    results[7] = calculate_coverage_summary(dataset["coverage_results"])
        
    path = f"./source_codes/correct_pyter_{model}_correct.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[8] = dataset["bug_detection"]

    path = f"./source_codes/correct_pyter_{model}_incorrect.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[9] = dataset["bug_detection"]

    path = f"./source_codes/total_pyter_{model}_correct.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[10] = dataset["bug_detection"]

    path = f"./source_codes/total_pyter_{model}_incorrect.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    results[11] = dataset["bug_detection"]

    humaneval = [results[0],results[1],results[2],results[3]]
    mbpp = [results[4],results[5],results[6],results[7]]
    apps = [results[8],results[9],results[10],results[11]]

    all_correct[model]["apps"] = apps
    all_correct[model]["mbpp"] = mbpp
    all_correct[model]["humaneval"] = humaneval


model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["humaneval","mbpp","apps",] 


for model in model_list:
    if "/" in model:
        model = model.split("/")[-1]
    humaneval_scores = [round(all_correct[model]['humaneval'][i], 2) for i in range(4)]
    mbpp_scores = [round(all_correct[model]['mbpp'][i], 2) for i in range(4)]
    apps_scores = [round(all_correct[model]['apps'][i], 2) for i in range(4)]
    
    max_humaneval_1 = max(humaneval_scores[0:2])
    max_humaneval_2 = max(humaneval_scores[2:4])
    max_mbpp_1 = max(mbpp_scores[0:2])
    max_mbpp_2 = max(mbpp_scores[2:4])
    max_apps_1 = max(apps_scores[0:2])
    max_apps_2 = max(apps_scores[2:4])
    
    humaneval_str = ' & '.join([f'\\textbf{{{score:.2f}}}' if score == max_humaneval_1 else str(score) for score in humaneval_scores[0:2]] + [f'\\textbf{{{score:.2f}}}' if score == max_humaneval_2 else str(score) for score in humaneval_scores[2:4]])
    mbpp_str = ' & '.join([f'\\textbf{{{score:.2f}}}' if score == max_mbpp_1 else str(score) for score in mbpp_scores[0:2]] + [f'\\textbf{{{score:.2f}}}' if score == max_mbpp_2 else str(score) for score in mbpp_scores[2:4]])
    apps_str = ' & '.join([f'\\textbf{{{score:.2f}}}' if score == max_apps_1 else str(score) for score in apps_scores[0:2]] + [f'\\textbf{{{score:.2f}}}' if score == max_apps_2 else str(score) for score in apps_scores[2:4]])

    print(f"{model} & {humaneval_str} & {mbpp_str} & {apps_str} \\\\")

overall = [0 for _ in range(12)]
idx = 0
for task in tasks:
    for prompt in range(4):
        for model in model_lists:
            if "/" in model:
                model = model.split("/")[-1]
            overall[idx]+=(all_correct[model][task][prompt])
        overall[idx] = round(overall[idx]/len(model_lists), 2)
        idx += 1

print("Overall & " + " & ".join([f"{round(score, 2)}" for score in overall]) + " \\\\")
