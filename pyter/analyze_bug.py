import ast
import json
def check_syntax(code):
    results = False
    try:
        ast.parse(code)
    except Exception as e:
        results = True
    return results

total_results = 0

with open(f"./evaluate.json", "r") as f:
    dataset = json.load(f)

for i in range(len(dataset)):
    results = check_syntax(dataset[i]["incorrect"])
    if results:
        total_results+=1

print(total_results)