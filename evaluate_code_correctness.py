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
correct_doctest = 0
correct_before_doctest = 0
correct_after_doctest = 0
result_original = 0
result_canonical_solution = 0
result_fuzzer = 0
result_fuzzer_canonical_solution = 0
idx_run_tests_orginal = []
idx_run_tests_canonical_solution = []
idx_run_tests_fuzzer = []
idx_run_tests_fuzzer_canonical_solution = []

language = ["python","cpp","js","go","js"]


class TimeoutException(Exception):
    pass
class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False
class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

# def process_humaneval_test(sample, problems, completion_code=False,test_case=False):
#     if test_case:
#         tests = sample["test_case"]
#     else:
#         tests = sample["test"]

#     if completion_code:
#         code = sample["completion"]
#     else:
#         code = sample["canonical_solution"]
#     if "```python" in code:
#         code = code[code.find("```python")+9:]
#         if "```" in code:
#             code = code[:code.find("```")]

#     import_pkgs = ""
#     if "test_imports" in sample.keys() and len(sample["test_imports"]) !=0:
#         for pkg in sample["test_imports"]:
#             import_pkgs += "\n"+pkg
    

#     without_code = import_pkgs + "\n" +code + "\n"
#     start_idx = len(without_code.split("\n"))

#     test_string = import_pkgs + "\n" +code + "\n" + tests


#     return test_string,start_idx


def process_humaneval_test(sample, problems, completion_code=True,test_case=False):

    test = sample["test"]
    if completion_code:
        code = sample["completion"]
    test_setup = ""
    if f"class {sample['entry_point']}" not in code: 
        test_string = test_setup + sample["prompt"] + code + "\n" + test + "\n"
    else:
        test_string = test_setup + code + "\n" + test + "\n"
    if "def check" in test_string:
        test_string += f"\ncheck({sample['entry_point']})"

    return test_string


def test_report(dataset, lg,task,model):
    coverage_results = {
        "canonical_solution_test_case":{},
        "completion_test": {},
        "completion_test_case": {},
    }
    failed_tests = {
        "completion_test": [],
        "completion_test_case": [],
    }
    correct_test_case = []
    correct = 0
    pass_1 = 0
    for item in tqdm(dataset):
        # test_setup = ""
        item["full_code"] = process_humaneval_test(item, dataset, completion_code=True,test_case=False)

        result = check_correctness(item["task_id"], item, lg, 5, "./tmp")
        if result["passed"]:
            item["passed"] = True
            correct += 1
        else:
            item["passed"] = False
        dataset[dataset.index(item)] = item
    print(round(correct/len(dataset)*100,2))
    return dataset


def fetch_completion(task,model):

    for prompt in ["prompt3"]:
        path = f"./results/update_{task}_{model}_{prompt}.json"
        with open(path,"r") as f:
            dataset = json.load(f)

        dataset = test_report(dataset,"python",task,model)
        with open(f"./results/update_{task}_{model}_{prompt}.json", "w") as f:
            json.dump(dataset,f,indent=4)


model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["apps","mbpp","humaneval",]
for model in model_list:
    if "/" in model:
        model = model.split("/")[1] 
    for task in tasks:
        print(f"Processing model: {model}")
        print(f"Processing task: {task}")

        fetch_completion(task, model)

