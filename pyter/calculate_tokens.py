import argparse
import os
import json
from concurrent.futures import ThreadPoolExecutor
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
def calculate_input_length(dataset):
    prompt1_tokens = 0
    prompt2_tokens = 0

    for i in range(len(dataset)):
        prompt1 = dataset[i]["correct"]
        prompt2 = dataset[i]["incorrect"]

        prompt1_tokens += len(encoding.encode(prompt1))
        prompt2_tokens += len(encoding.encode(prompt2))

    return prompt1_tokens, prompt2_tokens



with open(f"./evaluate.json", "r") as f:
    dataset = json.load(f)

prompt1_tokens, prompt2_tokens = calculate_input_length(dataset)
avg_prompt1_tokens = prompt1_tokens / len(dataset)
avg_prompt2_tokens = prompt2_tokens / len(dataset)

print(f"{avg_prompt1_tokens:.2f}&{avg_prompt2_tokens:.2f}&{len(dataset)}\\\\")