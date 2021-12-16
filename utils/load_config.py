import json
import os
import torch

absPath = os.path.join(os.path.dirname(__file__), "config.json")
with open(absPath) as file:
    config = json.load(file)

def get_attribute(name, defaultValue=None):
    try:
        return config[name]
    except KeyError:
        return defaultValue

config['device'] = f'cuda:{get_attribute("cuda")}' if torch.cuda.is_available() and get_attribute("cuda") >= 0 else 'cpu'
