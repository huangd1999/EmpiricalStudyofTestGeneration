import argparse
import os
import json
from tqdm import tqdm
import torch
import copy
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import random
from transformers import T5ForConditionalGeneration, AutoTokenizer,GPTNeoForCausalLM,AutoModelForCausalLM,AutoModel, AutoModelForSeq2SeqLM

model_list = ["meta-llama/Meta-Llama-3-8B","mistralai/Codestral-22B-v0.1","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","bigcode/starcoder2-7b"]
tasks = ["apps","mbpp", "humaneval"]

def preprocess_data(data,lg):
    if f"```{lg}" in data["test_case"]:
        data["test_case"] = data["test_case"][data["test_case"].find(f"```{lg}")+len(f"```{lg}"):]
        if "```" in data["test_case"]:
            data["test_case"] = data["test_case"][:data["test_case"].find("```")]
    else:
        if "```" in data["test_case"]:
            data["test_case"] = data["test_case"][data["test_case"].find("```")+3:]
            if "```" in data["test_case"]:
                data["test_case"] = data["test_case"][:data["test_case"].find("```")]
    return data


def batch_fetch_completion(dataset, model,text, batch_size=16,prompt_idx = "prompt1",completion = False):
    inputs_list = []
    for data_entry in dataset:
        if prompt == "prompt3":
            inputs = text + "\n### Input:\n"+ data_entry["prompt1"] + "\n" + data_entry["prompt3"] + data_entry["completion"] + "\n### TestCase:\n"
        if prompt == "prompt4":
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
        if prompt == "prompt5":
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
        inputs_list.append(inputs)

    batched_dataset = []
    for i in tqdm(range(0, len(inputs_list), batch_size)):
        batch_inputs = inputs_list[i:i+batch_size]
        try:
            encoded_inputs = tokenizer.batch_encode_plus(
                batch_inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024,
            ).to(model.device)
            
            outputs = model.generate(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                top_p=1,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, decoded_output in enumerate(decoded_outputs):
                data_entry = dataset[i+j].copy()
                data_entry["test_case"] = decoded_output
                if batch_inputs[j] in data_entry["test_case"]:
                    data_entry["test_case"] = data_entry["test_case"].replace(batch_inputs[j], "")
                data_entry = preprocess_data(data_entry, "python")
                batched_dataset.append(data_entry)

        except Exception as e:
            print(e)
            for j in range(len(batch_inputs)):
                data_entry = dataset[i+j].copy()
                data_entry["test_case"] = ""
                batched_dataset.append(data_entry)

    return batched_dataset

for checkpoint in model_list:
    print(checkpoint)
    name = checkpoint.split("/")[-1]
    if checkpoint in ["Salesforce/codet5p-220m-py", "Salesforce/codet5p-770m-py", "Salesforce/codet5p-2b", "Salesforce/codet5p-6b", "Salesforce/codet5p-16b"]:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint,device_map = "auto",trust_remote_code=True,torch_dtype = torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map = "auto",trust_remote_code=True, torch_dtype = torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    batch_size = 32

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    for dataset_name in tasks:
        with open(f"./prepare/{dataset_name}_update.json", "r") as f:
            dataset = json.load(f)
        dataset = [entry for entry in dataset]
        text = """
Please generate test cases for the uncomplete function to analyze whether the function is correct.
You should not use the test cases in the provided Input.
The test cases should be in the format of:
```python
assert function_name(input1) == expected_output1
assert function_name(input2) == expected_output2
```
"""

        dataset = batch_fetch_completion(dataset, model,text, batch_size=batch_size,prompt_idx = "prompt1",completion=False)

        with open(f"./new_prompt_codes/{dataset_name}_{name}_prompt1.json", "w") as f:
            json.dump(dataset, f, indent=4)

        dataset = batch_fetch_completion(dataset, model,text, batch_size=batch_size,prompt_idx = "prompt1",completion=False)

        with open(f"./rq3_codes/task_description_{dataset_name}_{name}.json", "w") as f:
            json.dump(dataset, f, indent=4)

        dataset = batch_fetch_completion(dataset, model,text, batch_size=batch_size,prompt_idx = "prompt1",completion=True)

        with open(f"./rq3_codes/completion_{dataset_name}_{name}.json", "w") as f:
            json.dump(dataset, f, indent=4)

        text = """
Please generate test cases for the completed function to analyze whether the function is correct.
You should not use the test cases in the provided Input.
The test cases should be in the format of:
```python
assert function_name(input1) == expected_output1
assert function_name(input2) == expected_output2
```
"""
        prompt_lists = ["prompt3","prompt4","prompt5"]

        for prompt in prompt_lists:

            dataset = batch_fetch_completion(dataset, model, text, batch_size=batch_size,prompt_idx = prompt)
            with open(f"./results/{dataset_name}_{name}_{prompt}.json", "w") as f:
                json.dump(dataset, f, indent=4)

