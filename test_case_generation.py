import argparse
import os
import json
from tqdm import tqdm
import copy
import time
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def preprocess_data(data,lg):
    if f"```{lg}" in data["test_case"]:
        data["test_case"] = data["test_case"][data["test_case"].find(f"```{lg}")+len(f"```{lg}"):]
        if "```" in data["test_case"]:
            data["test_case"] = data["test_case"][:data["test_case"].find("```")]
        data["test_case"] = data["test_case"][:data["test_case"].find("```")]
    else:
        if "```" in data["test_case"]:
            data["test_case"] = data["test_case"][data["test_case"].find("```")+3:]
            if "```" in data["test_case"]:
                data["test_case"] = data["test_case"][:data["test_case"].find("```")]
    return data

# Setting API parameters
openai.api_base = "https://api.aiohub.org/v1"
openai.api_key = 'API Key'



# Function to fetch completion
def uncomplete_fetch_completion(data_entry, model):
    text = """
Please generate test cases for the uncomplete function to analyze whether the function is correct.
You should not use the test cases in the provided Input.
The test cases should be in the format of:
```python
assert function_name(input1) == expected_output1
assert function_name(input2) == expected_output2
```
"""
    iterations = 0
    while True:
        iterations+=1
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                stream=False,
                messages=[
            {"role": "system", "content": "You are a code developer assistant."},
            {"role": "user", "content":text+"n### Input:\n"+data_entry["prompt1"] +f"\n### TestCase:\n"},
                ],
                request_timeout=100,
                top_p=1,
                temperature=0.0,
                # num_comps=1,
                max_tokens=1024,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            data_entry["test_case"] = completions.choices[0]["message"]["content"]
            data_entry = preprocess_data(data_entry,"python")
        except Exception as e:
            time.sleep(5)
            data_entry["test_case"] = "Assert RaiseError"
        if data_entry["test_case"] != "Assert RaiseError" or iterations>2:
            break
    return data_entry


def completed_fetch_completion(data_entry, model,prompt):
    text = """
Please generate test cases for the completed function to analyze whether the function is correct.
You should not use the test cases in the provided Input.
The test cases should be in the format of:
```python
assert function_name(input1) == expected_output1
assert function_name(input2) == expected_output2
```
"""
    if prompt == "prompt3":
        inputs = text + "\n### Input:\n"+ data_entry["prompt3"] + data_entry["completion"] + "\n### TestCase:\n"
    elif prompt == "prompt4":

        prompt4 = data_entry["prompt2"]
        if '"""' in prompt4:
            left = prompt4[:prompt4.find('"""')]
            right = prompt4[prompt4.rfind('"""'):]
            prompt4 = left + right
        if "'''" in prompt4:
            left = prompt4[:prompt4.find("'''")]
            right = prompt4[prompt4.rfind("'''"):]
            prompt4 = left + right
        inputs = text + "\n### Input:\n" +prompt4+ data_entry["completion"] + "\n### TestCase:\n"
    elif prompt == "prompt5":
        prompt5 = data_entry["prompt3"]
        if '"""' in prompt5:
            left = prompt5[:prompt5.find('"""')]
            right = prompt5[prompt5.rfind('"""'):]
            prompt5 = left + right
        if "'''" in prompt5:
            left = prompt5[:prompt5.find("'''")]
            right = prompt5[prompt5.rfind("'''"):]
            prompt5 = left + right
        inputs = text + "\n### Input:\n" +prompt5+ data_entry["completion"] + "\n### TestCase:\n"

    iterations = 0
    while True:
        iterations+=1
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                stream=False,
                messages=[
            {"role": "system", "content": "You are a code developer assistant."},
            {"role": "user", "content":inputs},
                ],
                request_timeout=100,
                top_p=1,
                temperature=0.0,
                # num_comps=1,
                max_tokens=1024,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            data_entry["test_case"] = completions.choices[0]["message"]["content"]
            data_entry = preprocess_data(data_entry,"python")

        except Exception as e:
            time.sleep(5)
            data_entry["test_case"] = "RaiseError"
        if data_entry["test_case"] != "RaiseError" or iterations>2:
            break
    return data_entry

model_list = ["gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["mbpp","humaneval","apps"]
for model in model_list:
    for dataset_name in tasks:
        with open(f"./prepare/{dataset_name}_update.json", "r") as f:
            dataset = json.load(f)
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_entry = {executor.submit(uncomplete_fetch_completion, copy.deepcopy(entry), model): entry for entry in tqdm(dataset)}
            for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
                entry = future_to_entry[future]
                try:
                    updated_entry = future.result()
                    idx = dataset.index(entry)
                    dataset[idx] = updated_entry
                except Exception as e:
                    print(repr(e))

        with open(f"./results/{dataset_name}_{model}_prompt1.json", "w") as f:
            json.dump(dataset, f, indent=4)

        for prompt in ["prompt3","prompt4","prompt5"]:
            with open(f"./prepare/{dataset_name}_update.json", "r") as f:
                dataset = json.load(f)

            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_entry = {executor.submit(completed_fetch_completion, copy.deepcopy(entry), model,prompt): entry for entry in tqdm(dataset)}
                for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
                    entry = future_to_entry[future]
                    try:
                        updated_entry = future.result()
                        idx = dataset.index(entry)
                        dataset[idx] = updated_entry
                    except Exception as e:
                        print(repr(e))

            with open(f"./results/{dataset_name}_{model}_{prompt}.json", "w") as f:
                json.dump(dataset, f, indent=4)
