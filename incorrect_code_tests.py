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
from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
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


def process_humaneval_test(sample, problems, test_case):
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    code = sample["prompt3"]

    test_string = test_setup + code + "\n" + test_case + "\n"

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
    total_correct = 0
    total_cases = 0
    for item in tqdm(dataset):
        lines = item["test_case"].split("\n")
        tests = []
        test_case = ""
        for line in lines:
            if "assert" in line:
                test_case += "\n" + line 
                tests.append(test_case)
                test_case = ""
            else:
                test_case += "\n" + line
        if test_case:  # add the last test case
            tests.append(test_case)

        correct_tests = []
        incorrect_tests = []
        for test_case in tests:
            item["full_code"] = process_humaneval_test(item, dataset, test_case=test_case)
            result = check_correctness(item["task_id"], item, lg, 5, "./tmp")
            if result["passed"]:
                correct_tests.append(test_case)
            else:
                incorrect_tests.append(test_case)
        correct += len(correct_tests)
        total_cases += len(tests)
        incorrect_tests = "\n".join(incorrect_tests)
        correct_tests = "\n".join(correct_tests)
        item["correct_tests"] = correct_tests
        item["incorrect_tests"] = incorrect_tests
        dataset[dataset.index(item)] = item

    print(round(correct/total_cases*100,2))
    return dataset



def fetch_completion(task,model):

    path = f"./results/{task}_{model}_prompt4.json"
    update_path = f"./results/prompt5_{task}_{model}_prompt4.json"
    with open(path,"r") as f:
        dataset = json.load(f)
    dataset = test_report(dataset,"python",task,model)
    with open(update_path, "w") as f:
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

