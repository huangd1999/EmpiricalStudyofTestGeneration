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
    
    if total == 0:
        return 0
    correctness = round(correct/total*100,2)
    return correctness



model_lists = ["gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku","meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1"]
tasks = ["humaneval","apps","mbpp"]
all_correct = {}
new_model_lists = []
all_correct = {}
for model in model_lists:
    if "/" in model:
        model = model.split("/")[1]
    new_model_lists.append(model)

    all_correct[model] = {}
    for task in tasks:
        correctness = []
        model_task_results = {}

        with open(f"./results/update_{task}_{model}_prompt2.json","r") as f:
            dataset2 = json.load(f)

        with open(f"./results/update_{task}_{model}_prompt3.json","r") as f:
            dataset3 = json.load(f)

        with open(f"./results/completion_{task}_{model}.json","r") as f:
            dataset = json.load(f)

        tmp_dataset = [[],[],[],[]]
        for i in range(len(dataset)):
            if dataset[i]["passed"]:
                tmp_dataset[0].append(dataset2[i])
                tmp_dataset[1].append(dataset[i])
            else:
                tmp_dataset[2].append(dataset3[i])
                tmp_dataset[3].append(dataset[i])
        
        for i in range(4):
            correctness.append(test_report(tmp_dataset[i]))
        
        # change format into table version.
        correctness[1],correctness[2] = correctness[2],correctness[1]
        all_correct[model][task] = correctness

model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["humaneval","mbpp","apps",] 



overall_gap1 = 0
overall_gap2 = 0
overall_gap3 = 0
overall_gap4 = 0
overall_gap5 = 0
overall_gap6 = 0
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
    
    if humaneval_scores[0] != 0:
        humaneval_gap = round((humaneval_scores[0]-humaneval_scores[1])/humaneval_scores[0]*100, 2)
    else:
        humaneval_gap = 0
    
    if humaneval_scores[2] != 0:
        humaneval_gap2 = round((humaneval_scores[2]-humaneval_scores[3])/humaneval_scores[2]*100, 2)
    else:
        humaneval_gap2 = 0
    
    if mbpp_scores[0] != 0:
        mbpp_gap = round((mbpp_scores[0]-mbpp_scores[1])/mbpp_scores[0]*100, 2)
    else:
        mbpp_gap = 0
    
    if mbpp_scores[2] != 0:
        mbpp_gap2 = round((mbpp_scores[2]-mbpp_scores[3])/mbpp_scores[2]*100, 2)
    else:
        mbpp_gap2 = 0
    
    if apps_scores[0] != 0:
        apps_gap = round((apps_scores[0]-apps_scores[1])/apps_scores[0]*100, 2)
    else:
        apps_gap = 0
    
    if apps_scores[2] != 0:
        apps_gap2 = round((apps_scores[2]-apps_scores[3])/apps_scores[2]*100, 2)
    else:
        apps_gap2 = 0
    
    print(f"{model} & {humaneval_scores[0]-humaneval_scores[1]:.2f}&{humaneval_scores[2]-humaneval_scores[3]:.2f} & {humaneval_gap:.2f}  & {humaneval_gap2:.2f}& {mbpp_scores[0]-mbpp_scores[1]:.2f} & {mbpp_scores[2]-mbpp_scores[3]:.2f} & {mbpp_gap:.2f} & {mbpp_gap2:.2f} & {apps_scores[0]-apps_scores[1]:.2f}& {apps_scores[2]-apps_scores[3]:.2f} & {apps_gap:.2f} & {apps_gap2:.2f}\\\\")
    if humaneval_scores[0]!=0:
        overall_gap1+=(humaneval_scores[0]-humaneval_scores[1])/humaneval_scores[0]
    else:
        overall_gap1+=0
    if humaneval_scores[2]!=0:
        overall_gap2+=(humaneval_scores[2]-humaneval_scores[3])/humaneval_scores[2]
    else:
        overall_gap2+=0
    if mbpp_scores[0]!=0:
        overall_gap3+=(mbpp_scores[0]-mbpp_scores[1])/mbpp_scores[0]
    else:
        overall_gap3+=0
    if mbpp_scores[2]!=0:
        overall_gap4+=(mbpp_scores[2]-mbpp_scores[3])/mbpp_scores[2]
    else:
        overall_gap4+=0
    if apps_scores[0]!=0:
        overall_gap5+=(apps_scores[0]-apps_scores[1])/apps_scores[0]
    else:
        overall_gap5+=0
    if apps_scores[2]!=0:
        overall_gap6+=(apps_scores[2]-apps_scores[3])/apps_scores[2]
    else:
        overall_gap6+=0

overall_gap1 = overall_gap1/len(model_list)*100
overall_gap2 = overall_gap2/len(model_list)*100
overall_gap3 = overall_gap3/len(model_list)*100
overall_gap4 = overall_gap4/len(model_list)*100
overall_gap5 = overall_gap5/len(model_list)*100
overall_gap6 = overall_gap6/len(model_list)*100


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


print(f"Overall & {overall[0]-overall[1]:.2f} & {overall[2]-overall[3]:.2f}  & {overall_gap1:.2f}& {overall_gap2:.2f} & {overall[4]-overall[5]:.2f} & {overall[6]-overall[7]:.2f} & {overall_gap3:.2f} & {overall_gap4:.2f}& {overall[8]-overall[9]:.2f}& {overall[10]-overall[11]:.2f}& {overall_gap5:.2f} & {overall_gap6:.2f} \\\\")
