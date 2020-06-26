import json
import os

absPath = os.path.join(os.path.dirname(__file__), "config.json")
with open(absPath) as file:
    config = json.load(file)

def get_attribute(name, defaultValue=None):
    try:
        return config[name]
    except KeyError:
        return defaultValue