
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


def test_report(dataset):

    correct = 0
    total = 0
    for i in range(len(dataset)):
        correct_test = dataset[i]["correct_tests"]
        total_cases = dataset[i]["correct_tests"] + "\n" + dataset[i]["incorrect_tests"]
        correct += len([line for line in correct_test.split("\n") if "assert" in line])
        total += len([line for line in total_cases.split("\n") if "assert" in line])
    
    correctness = round(correct/total*100,2)
    return correctness


def task_report(dataset):

    correct = 0
    total = 0
    for i in range(len(dataset)):
        if "assert" not in dataset[i]["incorrect_tests"]:
            correct+=1
        total+=1
    
    correctness = round(correct/total*100,2)
    return correctness


model_lists = ["gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku","meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1"]

tasks = ["humaneval","apps","mbpp"]
all_correct = {}
new_model_lists = []
all_correct = {}
for model in model_lists:
    if "/" in model:
        model = model.split("/")[-1]
    new_model_lists.append(model)

    all_correct[model] = {}
    for task in tasks:
        correctness = []
        task_level = []
        model_task_results = {}
        for step in range(4,6):
            with open(f"./results/prompt5_{task}_{model}_prompt{step}.json", "r") as f:
                dataset = json.load(f)
            result = test_report(dataset)
            correctness.append(result)
            result = task_report(dataset)
            task_level.append(result)
        correctness = correctness+task_level
        print(correctness)
        all_correct[model][task] = correctness







model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["humaneval","mbpp","apps",] 


for model in model_list:
    if "/" in model:
        model = model.split("/")[1]
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
                model = model.split("/")[1]
            overall[idx]+=(all_correct[model][task][prompt])
        overall[idx] = round(overall[idx]/len(model_lists), 2)
        idx += 1
    # break

print("Overall & " + " & ".join([f"{round(score, 2)}" for score in overall]) + " \\\\")