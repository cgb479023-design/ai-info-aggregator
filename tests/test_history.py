import json
from pathlib import Path

from src.history import load_history


def test_load_history_returns_empty_list_when_file_missing(tmp_path):
    missing = tmp_path / "nope.json"
    assert load_history(str(missing)) == []


def test_load_history_returns_pushed_list_when_file_exists(tmp_path):
    path = tmp_path / "pushed.json"
    data = {
        "pushed": [
            {"url": "https://a.com", "pushed_at": "2026-04-01", "title": "A", "score": 8}
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_history(str(path))
    assert result == data["pushed"]


def test_load_history_returns_empty_list_when_file_is_malformed(tmp_path):
    path = tmp_path / "pushed.json"
    path.write_text("{ not valid json", encoding="utf-8")
    assert load_history(str(path)) == []
