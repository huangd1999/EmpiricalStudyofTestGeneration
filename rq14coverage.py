import random
import json
import io
import inspect
import numpy as np
import re
import sys
sys.path.append('./CodeGeeX/')
import contextlib
import faulthandler
from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
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

def process_humaneval_test(sample, problems, completion_code=True,test_case=False):

    if not test_case:
        test = sample["test"]
    else:
        test = sample["correct_tests"] + "\n" + sample["incorrect_tests"]
    if completion_code:
        # code = sample["completion"]
        code = sample["prompt3"]
    test_setup = ""
    if f"class {sample['entry_point']}" not in code: 
        test_string = test_setup + sample["prompt"] + code + "\n" + test + "\n"
    else:
        test_string = test_setup + code + "\n" + test + "\n"
    if "def check" in test_string:
        test_string += f"\ncheck({sample['entry_point']})"

    return test_string



import random

def test_report(dataset, lg,task,model,step):

    bug_detection = 0
    for item in tqdm(dataset):
        if "assert" in item["incorrect_tests"]:
            continue
        item["full_code"]= process_humaneval_test(item, dataset, completion_code=True,test_case=True)
        result = check_correctness(item["task_id"], item, lg, 5, "./tmp")
        if not result["passed"]:
            item["full_code"]= process_humaneval_test(item, dataset, completion_code=True,test_case=False)
            tmp_result = check_correctness(item["task_id"], item, lg, 5, "./tmp")
            if not tmp_result["passed"]:
                # if random.random()<0.1 and step==5:
                #     continue
                bug_detection+=1

    bug_detection = bug_detection/len(dataset)*100
    result = {
        "bug_detection":bug_detection,
    }
    print(bug_detection)

    return result
def fetch_completion(task,model):
    for step in range(1,6):
        path = f"./results/update_{task}_{model}_prompt{step}.json"
        with open(path,"r") as f:
            dataset = json.load(f)

        with open(f"./results/incorrect_completion_{task}_gpt-3.5-turbo-1106.json","r") as f:
            tmp_dataset = json.load(f)
        for i in range(len(dataset)):
            dataset[i]["completion"] = tmp_dataset[i]["completion"]
            dataset[i]["test"] = tmp_dataset[i]["test"]
            dataset[i]["prompt4"] = tmp_dataset[i]["completion"]

        result = test_report(dataset,"python",task,model,step)

        with open(f"./results/prompt3_total_{task}_{model}_prompt{step}.json","w") as f:
            json.dump(result,f,indent=4)


model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["humaneval","apps","mbpp"]

for model in model_list:
    for task in tasks:
        if "/" in model:
            model = model.split("/")[1]
        print(f"Processing model: {model} | task: {task}")

        fetch_completion(task, model)

