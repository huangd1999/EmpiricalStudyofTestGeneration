import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import kendalltau, spearmanr


def load_correctness_data(model_list, tasks):
    all_correct = {}
    for model in model_list:
        model_name = model.split("/")[-1]
        all_correct[model_name] = {}
        for task in tasks:
            correctness = []
            for step in range(1, 6):
                with open(f"./results/update_{task}_{model_name}_prompt{step}.json", "r") as f:
                    dataset = json.load(f)
                correctness.append(test_report(dataset))
            all_correct[model_name][task] = correctness
    return all_correct

def test_report(dataset):
    correct = sum(len([line for line in entry["incorrect_tests"].split("\n") if "assert" in line]) for entry in dataset)
    total = sum(len([line for line in (entry["correct_tests"] + "\n" + entry["incorrect_tests"]).split("\n") if "assert" in line]) for entry in dataset)
    return round(correct / total * 100, 2)

def load_passed_data(model_list, tasks):
    correctness_list = {}
    for model in model_list:
        model_name = model.split("/")[-1]
        correctness_list[model_name] = {}
        for task in tasks:
            with open(f"./rq3_codes/{task}_{model_name}.json", "r") as f:
                dataset = json.load(f)
            correctness_list[model_name][task] = len([entry for entry in dataset if entry["passed"]]) / len(dataset) * 100
    return correctness_list



def main():
    model_list = [
        "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-turbo-preview", "gpt-4", 
        "claude-3-sonnet", "claude-3-haiku", "meta-llama/Meta-Llama-3-8B", 
        "meta-llama/CodeLlama-7b-Python-hf", "deepseek-ai/deepseek-coder-6.7b-instruct", 
        "starcoder2-7b", "Codestral-22B-v0.1"
    ]
    tasks = ["humaneval","mbpp","apps"]

    all_correct = load_correctness_data(model_list, tasks)
    correctness_list = load_passed_data(model_list, tasks)

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(5, 3, figsize=(24, 18), sharey=False)


    for row, step in enumerate(range(1, 6)):
        corrs = []
        p_values = []
        for col, task in enumerate(tasks):
            x = [correctness_list[model.split("/")[-1]][task] for model in model_list]
            y = [all_correct[model.split("/")[-1]][task][step - 1] for model in model_list]
            corr, p_value = kendalltau(x, y)

            p_values.append(p_value)
            corrs.append(corr)

        print(f"&{round(corrs[0],2)} ({round(p_values[0],2)})&{round(corrs[1],2)} ({round(p_values[1],2)})&{round(corrs[2],2)} ({round(p_values[2],2)})\\\\")

    print()
if __name__ == "__main__":
    main()
