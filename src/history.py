import json
import os


def load_history(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("pushed", [])
    except (json.JSONDecodeError, OSError):
        return []
