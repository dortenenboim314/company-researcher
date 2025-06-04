import os

BASE_PROMPT_DIR = os.path.dirname(__file__)

def load_prompt(filename: str) -> str:
    path = os.path.join(BASE_PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()