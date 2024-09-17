import argparse
import os
import json
from tqdm import tqdm
import copy
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datasets import load_dataset
import time


def preprocess_data(data):
    if f"```python" in data["completion"]:
        data["completion"] = data["completion"][data["completion"].find(f"```python")+len(f"```python"):]
        if "```" in data["completion"]:
            data["completion"] = data["completion"][:data["completion"].find("```")]
    else:
        if "```" in data["completion"]:
            data["completion"] = data["completion"][data["completion"].find("```")+3:]
            if "```" in data["completion"]:
                data["completion"] = data["completion"][:data["completion"].find("```")]
    return data


# Function to fetch completion
def fetch_completion(data_entry, model):
    text = """
I am the Python course teacher. I plan to create an incorrect code and then require students debug it.
Can you then based on the below task description to write a incorrect code?
You should write Python function in ```python\n[Code]\n``` format.
Please replace [Code] with your code.
"""
    reexecutetimes = 0
    while True:
        reexecutetimes+=1
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                stream=False,
                messages=[
            {"role": "system", "content": "You are a code developer."},
            {"role": "user", "content":text+f"\n```python\n{data_entry['prompt']}\n```"},
                ],
                request_timeout=100,
                top_p=1,
                temperature=0.0,
                # num_comps=1,
                max_tokens=1024,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            data_entry["completion"] = completions.choices[0]["message"]["content"]
            data_entry = preprocess_data(data_entry)
        except Exception as e:
            print(repr(e))
            time.sleep(20)
            data_entry["completion"] = ""
        if data_entry["completion"]!="" or reexecutetimes>5:
            break
    return data_entry

if __name__ == "__main__":
    model_list = ["gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4"]
    for dataset_name in ["apps","humaneval","mbpp"]:
        for model in model_list:
            with open(f"./prepare/{dataset_name}.json", "r") as f:
                dataset = json.load(f)
            dataset = [entry for entry in dataset]
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry), model): entry for entry in tqdm(dataset)}
                for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
                    entry = future_to_entry[future]
                    try:
                        updated_entry = future.result()
                        idx = dataset.index(entry)
                        dataset[idx] = updated_entry
                    except Exception as e:
                        print(repr(e))

            with open(f"./codes/incorrect_completion_{dataset_name}_{model}.json", "w") as f:
                json.dump(dataset, f, indent=4)
