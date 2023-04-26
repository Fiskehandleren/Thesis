import subprocess

import json
import re

def remove_comments(content):
    return re.sub(r'//.*', '', content)

def load_launch_json(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        content_no_comments = remove_comments(content)
        data = json.loads(content_no_comments)
    return data


def run_main_py(parameters) -> bool:
    command = ['python', 'main.py']
    command.extend(parameters)
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print(f"{parameters[1]}: ❌")
        return False
    else:
        print(f"{parameters[1]}: ✅")
        return True


if __name__ == '__main__':
    configurations = []
    # Read configs from .vscode/launch.json
    vscode_launch_file_path = '.vscode/launch.json'
    launch_data = load_launch_json(vscode_launch_file_path)
    
    models = ["TGCN", "LSTM", "GRU", "AR", "ARNet"]
    for config in launch_data["configurations"]:
        if config['name'] in models:
            configurations.append(config["args"])

    runs = []
    for config in configurations:
        runs.append(run_main_py(config))
    if all(runs):
        exit(0)
    else:
        exit(1)