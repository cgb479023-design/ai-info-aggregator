import json
from pathlib import Path

from src.history import filter_unseen, load_history


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


def test_filter_unseen_keeps_all_when_history_empty():
    articles = [{"url": "https://a.com", "title": "A"}]
    unseen, skipped = filter_unseen(articles, history=[], days=90, today="2026-04-21")
    assert unseen == articles
    assert skipped == []


def test_filter_unseen_skips_recent_match():
    articles = [{"url": "https://a.com", "title": "A"}]
    history = [{"url": "https://a.com", "pushed_at": "2026-04-15", "title": "A", "score": 8}]
    unseen, skipped = filter_unseen(articles, history, days=90, today="2026-04-21")
    assert unseen == []
    assert skipped == articles


def test_filter_unseen_ignores_expired_match():
    articles = [{"url": "https://a.com", "title": "A"}]
    history = [{"url": "https://a.com", "pushed_at": "2025-12-01", "title": "A", "score": 8}]
    unseen, skipped = filter_unseen(articles, history, days=90, today="2026-04-21")
    assert unseen == articles
    assert skipped == []


def test_filter_unseen_exact_boundary_is_still_within_window():
    # 90 天前 == 窗口内
    articles = [{"url": "https://a.com", "title": "A"}]
    history = [{"url": "https://a.com", "pushed_at": "2026-01-21", "title": "A", "score": 8}]
    unseen, skipped = filter_unseen(articles, history, days=90, today="2026-04-21")
    assert skipped == articles


def test_filter_unseen_handles_mix():
    articles = [
        {"url": "https://a.com", "title": "A"},
        {"url": "https://b.com", "title": "B"},
        {"url": "https://c.com", "title": "C"},
    ]
    history = [
        {"url": "https://a.com", "pushed_at": "2026-04-10", "title": "A", "score": 8},
        {"url": "https://c.com", "pushed_at": "2025-01-01", "title": "C", "score": 7},
    ]
    unseen, skipped = filter_unseen(articles, history, days=90, today="2026-04-21")
    unseen_urls = {a["url"] for a in unseen}
    skipped_urls = {a["url"] for a in skipped}
    assert unseen_urls == {"https://b.com", "https://c.com"}
    assert skipped_urls == {"https://a.com"}
