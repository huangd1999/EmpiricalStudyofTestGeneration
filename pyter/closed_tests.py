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
def uncomplete_fetch_completion(data_entry, model,completion="correct"):
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
            {"role": "user", "content":text+"n### Input:\n"+data_entry[completion] + f"\n### TestCase:\n"},
                ],
                request_timeout=100,
                top_p=1,
                temperature=0.0,
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



model_list = ["gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
for model in model_list:
    
    with open(f"./evaluate.json", "r") as f:
        dataset = json.load(f)
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_entry = {executor.submit(uncomplete_fetch_completion, copy.deepcopy(entry), model,completion="incorrect"): entry for entry in tqdm(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))

    with open(f"./source_codes/incorrect_pyter_{model}.json", "w") as f:
        json.dump(dataset, f, indent=4)

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_entry = {executor.submit(uncomplete_fetch_completion, copy.deepcopy(entry), model,completion="correct"): entry for entry in tqdm(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))

    with open(f"./source_codes/correct_pyter_{model}.json", "w") as f:
        json.dump(dataset, f, indent=4)

