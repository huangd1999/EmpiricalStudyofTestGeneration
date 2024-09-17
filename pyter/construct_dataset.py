import os
import json

needed_files = ["youtubedl-11"," scrapy-29", "pandas-142", "luigi-26", "keras-39", "Zappa-388", "sanic-2008-2", "sanic-2008-1", "salt-52624", "salt-38908"]

def get_file_contents(folder_path):
    file_contents_dict = []
    exists = []
    for root, dirs, files in os.walk(folder_path):
        py_files = [file for file in files if file.endswith(".py")]
        for py_file in py_files:
            prefix = py_file[:-3]  # 去除文件扩展名
            py_solution_file = prefix + ".py_solution"
            
            if py_solution_file in files:
                py_file_path = os.path.join(root, py_file)
                py_solution_file_path = os.path.join(root, py_solution_file)
                
                with open(py_file_path, "r") as f:
                    py_file_content = f.read()
                
                with open(py_solution_file_path, "r") as f:
                    py_solution_file_content = f.read()
                
                file_name = py_file_path.split("/")[-1].split(".")[0]
                end = file_name.split("-")[-1]
                current_folder = os.path.basename(root)  # 获取当前文件夹名字
                # print(f"当前文件夹: {current_folder}")
                # if current_folder  not in needed_files:
                #     continue
                print(f"当前文件夹: {current_folder}")
                exists.append(current_folder)
                file_contents_dict.append({
                    "py_file_path": file_name,
                    "incorrect": py_file_content,
                    "correct": py_solution_file_content
                })
        # print([entry for entry in needed_files if entry not in exists])
    
    return file_contents_dict

# 指定evaluate文件夹的路径
evaluate_folder = "./PyTER/evaluate"

# 获取文件内容字典
file_contents_dict = get_file_contents(evaluate_folder)
print(len(file_contents_dict))

with open("./evaluate.json", "w") as f:
    json.dump(file_contents_dict, f, indent=4)
