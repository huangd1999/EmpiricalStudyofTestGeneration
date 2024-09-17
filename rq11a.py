
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



model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["humaneval","mbpp","apps",] 
all_correct = {}
new_model_lists = []
all_correct = {}
for model in model_list:
    if "/" in model:
        model = model.split("/")[1]
    new_model_lists.append(model)

    all_correct[model] = {}
    for task in tasks:
        correctness = []
        model_task_results = {}
        for step in range(1,6):
            with open(f"./results/rq11a_12a_{task}_{model}_prompt{step}.json", "r") as f:
                dataset = json.load(f)
        
            correctness.append(dataset["correctness"])
        all_correct[model][task] = correctness


for model in model_list:
    if "/" in model:
        model = model.split("/")[1]
    humaneval_scores = [round(all_correct[model]['humaneval'][i], 2) for i in range(5)]
    mbpp_scores = [round(all_correct[model]['mbpp'][i], 2) for i in range(5)]
    apps_scores = [round(all_correct[model]['apps'][i], 2) for i in range(5)]
    
    max_humaneval = max(humaneval_scores)
    max_mbpp = max(mbpp_scores)
    max_apps = max(apps_scores)
    
    humaneval_str = ' & '.join([f'\\textbf{{{score:.2f}}}' if score == max_humaneval else str(score) for score in humaneval_scores])
    mbpp_str = ' & '.join([f'\\textbf{{{score:.2f}}}' if score == max_mbpp else str(score) for score in mbpp_scores])
    apps_str = ' & '.join([f'\\textbf{{{score:.2f}}}' if score == max_apps else str(score) for score in apps_scores])
    
    print(f"{model} & {humaneval_str} & {mbpp_str} & {apps_str} \\\\")


overall = [0 for _ in range(15)]
idx = 0
for task in tasks:
    for prompt in range(5):
        for model in model_list:
            if "/" in model:
                model = model.split("/")[1]
            overall[idx]+=(all_correct[model][task][prompt])
        overall[idx] = round(overall[idx]/len(model_list), 2)
        idx += 1

print("Overall & " + " & ".join([f"{round(score, 2)}" for score in overall]) + " \\\\")

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

tasks_labels = ["humaneval", "mbpp", "apps"]
colors = ['#FF9999', '#66B2FF', '#99FF99']

for idx, task in enumerate(tasks_labels):
    prompt1 = [all_correct[model.split("/")[1]][task][0] if "/" in model else all_correct[model][task][0] for model in model_list]
    prompt2 = [all_correct[model.split("/")[1]][task][1] if "/" in model else all_correct[model][task][1] for model in model_list]
    prompt3 = [all_correct[model.split("/")[1]][task][2] if "/" in model else all_correct[model][task][2] for model in model_list]
    prompt4 = [all_correct[model.split("/")[1]][task][3] if "/" in model else all_correct[model][task][3] for model in model_list]
    prompt5 = [all_correct[model.split("/")[1]][task][4] if "/" in model else all_correct[model][task][4] for model in model_list]
    data = [prompt1, prompt2, prompt3, prompt4, prompt5]


    bplot = axs[idx].boxplot(data, patch_artist=True)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    for median in bplot['medians']:
        median.set(color='#FF0000', linewidth=2)


    axs[idx].set_xticklabels(["P_T", "P_T_CC", "P_T_IC", "P_CC", "P_IC"], fontsize=30,rotation=90)
    axs[idx].set_title(task, fontsize=30)
    axs[idx].set_ylabel('Correctness (%)', fontsize=30)
    axs[idx].tick_params(axis='y',labelsize=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./figures/rq11a.pdf', dpi=300)
plt.show()
