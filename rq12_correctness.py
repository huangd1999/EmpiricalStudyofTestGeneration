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

def process_humaneval_test(sample, problems, completion_code=False,test_case=False):

    test = sample["correct_tests"] + "\n" + sample["incorrect_tests"]

    code = sample["canonical_solution"]

    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"

    test_string = test_setup + sample["prompt"] + code + "\n" + test + "\n"

    if "def check" in test_string:
        test_string += f"\ncheck({sample['entry_point']})"

    helper_string = "\n" + test + "\n" + f"\ncheck({sample['entry_point']})"

    start_idx = len(test_string.split("\n")) - len(helper_string.split("\n"))

    return test_string, start_idx


def calculate_coverage(entry_point,task,model):
    timeout = 5.0
    full_code = entry_point["full_code"]
    start_idx = entry_point["start_idx"]

    if "/" in entry_point["task_id"]:
        temp_file_name = f"./tmp/{task}_{model}"+entry_point["task_id"].split("/")[1]+".py"
    else: 
        temp_file_name = f"./tmp/{task}_{model}"+entry_point["task_id"]+".py"

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    with open(temp_file_name,"w") as temp_file:
        temp_file.write(full_code)

    stmts, miss, cover, missing_lines = 0, 0, 0, []

    try:
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                subprocess.run(["coverage", "run", temp_file_name], capture_output=True, text=True, check=False)
        result = subprocess.run(["coverage", "report", "-m", temp_file_name], capture_output=True, text=True, check=True)

        pattern = re.compile(rf"{re.escape(os.path.basename(temp_file_name))}\s+(\d+)\s+(\d+)\s+(\d+)%\s+([\d\-,\s]+)")
        match = pattern.search(result.stdout)

        if match:
            stmts, miss, cover = map(int, match.groups()[:3])
            missing = match.group(4)
            missing_lines_str = re.findall(r"\d+(?:-\d+)?", missing)
            missing_lines = []
            for range_str in missing_lines_str:
                if "-" in range_str:
                    start, end = map(int, range_str.split("-"))
                    missing_lines.extend(range(start, end + 1))
                else:
                    missing_lines.append(int(range_str))
        start_idx = entry_point["start_idx"]
        need_remove_lines = []
        miss = 0
        for line in missing_lines:
            if line>=start_idx:
                need_remove_lines.append(line)
            else:
                miss += 1
        stmts = start_idx
    except: 
        os.remove(temp_file_name)
        import_pkgs = ""
        if "test_imports" in entry_point.keys() and len(entry_point["test_imports"]) !=0:
            for pkg in entry_point["test_imports"]:
                import_pkgs += "\n"+pkg
        full_code = import_pkgs + "\n" + entry_point["canonical_solution"]
        if "/" in entry_point["task_id"]:
            temp_file_name = f"./tmp/{task}_{model}"+entry_point["task_id"].split("/")[1]+".py"
        else: 
            temp_file_name = f"./tmp/{task}_{model}"+entry_point["task_id"]+".py"

        with open(temp_file_name,"w") as temp_file:
            temp_file.write(full_code)
        subprocess.run(["coverage", "run", temp_file_name], check=True, capture_output=True)
        tmp = subprocess.run(["coverage", "report", temp_file_name], capture_output=True, text=True, check=True)
        pattern = re.compile(rf"{re.escape(os.path.basename(temp_file_name))}\s+(\d+)\s+(\d+)\s+(\d+)%")
        match = pattern.search(tmp.stdout)

        if match:
            stmts, miss, cover = map(int, match.groups())
        os.remove(temp_file_name)

    return stmts, miss, cover

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
        if "dataset" in item.keys() and item["dataset"] =="humaneval":
            item["canonical_solution"] = item["prompt"] + "\n" + item["canonical_solution"]
        if task == "humaneval":
            item["canonical_solution"] = item["prompt"] + "\n" + item["canonical_solution"]

        item["test_case"] = item["correct_tests"] + "\n" + item["incorrect_tests"]

        if len([test for test in item["test_case"].split("/") if "assert" in test]) == 0:
             item["test_case"] = "assert RaisesError"
        item["full_code"],item["start_idx"] = process_humaneval_test(item, dataset, completion_code=False,test_case=True)
        result = check_correctness(item["task_id"], item, lg, 5, "./tmp")
        if result["passed"]:
            correct += 1
            stmts, miss, cover = calculate_coverage(item,task,model)
            correct_test_case.append(item["task_id"])
            coverage_results["canonical_solution_test_case"][item["task_id"]] = {"stmts": stmts, "miss": miss, "cover": cover,"pass":result["passed"]}
        else:
            entry_point = item
            import_pkgs = ""
            if "test_imports" in entry_point.keys() and len(entry_point["test_imports"]) !=0:
                for pkg in entry_point["test_imports"]:
                    import_pkgs += "\n"+pkg
            full_code = import_pkgs + "\n" + entry_point["canonical_solution"]
            if "/" in entry_point["task_id"]:
                temp_file_name = f"./tmp/{task}_{model}"+entry_point["task_id"].split("/")[1]+".py"
            else: 
                temp_file_name = f"./tmp/{task}_{model}"+entry_point["task_id"]+".py"

            with open(temp_file_name,"w") as temp_file:
                temp_file.write(full_code)
            subprocess.run(["coverage", "run", temp_file_name], check=True, capture_output=True)
            tmp = subprocess.run(["coverage", "report", temp_file_name], capture_output=True, text=True, check=True)
            pattern = re.compile(rf"{re.escape(os.path.basename(temp_file_name))}\s+(\d+)\s+(\d+)\s+(\d+)%")
            match = pattern.search(tmp.stdout)

            if match:
                stmts, miss, cover = map(int, match.groups())
            os.remove(temp_file_name)

            coverage_results["canonical_solution_test_case"][item["task_id"]] = {"stmts": stmts, "miss": miss, "cover": cover,"pass":result["passed"]}


    correctness = round(correct/len(dataset)*100,2)

    result = {
        "coverage_results":coverage_results,
        "correctness":correctness,
    }

    print(f"Correctness: {correctness}")
    return result


def fetch_completion(task,model):
    for step in range(1,4):
        path = f"./results/update_{task}_{model}_prompt{step}.json"
        with open(path,"r") as f:
            dataset = json.load(f)

        result = test_report(dataset,"python",task,model)

        with open(f"./results/rq11b_12b_{task}_{model}_prompt{step}.json","w") as f:
            json.dump(result,f,indent=4)


model_list = ["meta-llama/Meta-Llama-3-8B","meta-llama/CodeLlama-7b-Python-hf","deepseek-ai/deepseek-coder-6.7b-instruct","starcoder2-7b","Codestral-22B-v0.1","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-sonnet","claude-3-haiku"]
tasks = ["apps","mbpp","humaneval",]
prompts = ["prompt1","prompt2","prompt3"]
for model in model_list:
    if "/" in model:
        model = model.split("/")[1] 
    for task in tasks:       
        print(f"Processing model: {model}")
        print(f"Processing task: {task}")

        fetch_completion(task, model)


