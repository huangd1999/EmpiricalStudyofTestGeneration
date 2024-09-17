import random
import json
from typing import Optional, Callable, Dict
import ast
import doctest
import subprocess
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import numpy as np
import re
import sys
sys.path.append('./CodeGeeX/')
import contextlib
import faulthandler
import io
import os
import coverage
import multiprocessing
import platform
import signal
from tqdm import tqdm
from codegeex.benchmark.execution import check_correctness
import tempfile
import io
import sys
import doctest
import random

model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1"]
tasks = ["apps","mbpp","humaneval"]

for task in tasks:
    print(f"Processing task: {task}")
    

    all_data = []
    for model in model_list:
        if "/" in model:
            model = model.split("/")[1]
        path = f"./results/{task}_{model}.json"

        with open(path, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    data_by_task = {}
    for item in all_data:
        task_id = item["task_id"]
        if task_id not in data_by_task:
            data_by_task[task_id] = []
        data_by_task[task_id].append(item)

    updated_data = []
    for task_id, task_data in data_by_task.items():
        error_codes = [item for item in task_data if not item.get("passed", False)]
        if error_codes:
            selected_code = random.choice(error_codes)
            incorrect_completion = selected_code["completion"]
        else:
            selected_code = random.choice(task_data)
            incorrect_completion = selected_code["completion"].split("\n")
            incorrect_completion = "\n".join([line for line in incorrect_completion if "return" not in line])
        
        base_data = random.choice(task_data)
        base_data["prompt1"] = base_data["prompt"]
        base_data["prompt2"] = base_data["prompt"] + base_data["canonical_solution"]
        base_data["prompt3"] = base_data["prompt"] + base_data["incorrect_completion"]
        updated_data.append(base_data)

    output_path = f"./prepare/{task}_update.json"
    with open(output_path, "w") as f:
        json.dump(updated_data, f, indent=4)