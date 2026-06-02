import pytest

from agent import loop

DB = "sqlite-data/concert_singer/concert_singer.sqlite"


def test_loop_recovers_from_a_retryable_error(monkeypatch):
    sqls = iter(["SELECT nope FROM singer", "SELECT Name FROM singer"])
    monkeypatch.setattr(loop, "generate_sql", lambda *a, **k: next(sqls))
    monkeypatch.setattr(loop, "_regenerate", lambda *a, **k: next(sqls))
    assert loop.solve_sql("names of singers", DB) == "SELECT Name FROM singer"


def test_loop_returns_first_attempt_when_it_runs(monkeypatch):
    monkeypatch.setattr(loop, "generate_sql", lambda *a, **k: "SELECT Name FROM singer")
    monkeypatch.setattr(loop, "_regenerate", lambda *a, **k: pytest.fail("no retry"))
    assert loop.solve_sql("names of singers", DB) == "SELECT Name FROM singer"
