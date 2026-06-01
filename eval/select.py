"""Sample a curated eval set from Spider's dev.json into eval/spider_cases.jsonl."""

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

DATA = Path("sqlite-data")
DEV = DATA / "dev.json"
OUT = Path("eval/spider_cases.jsonl")
EXCLUDE = {"wta_1"}  # 100 MB; gitignored, so no case may depend on it
PER_DB = 3
SEED = 0


def difficulty(sql: str) -> str:
    """Rough label from SQL shape — a starting point, not Spider's official metric."""
    s = sql.lower()
    if " union " in s or " intersect " in s or " except " in s or s.count("select") > 1:
        return "extra"
    joins = s.count(" join ")
    agg = any(f"{fn}(" in s for fn in ("count", "sum", "avg", "min", "max"))
    if joins >= 2 or ("group by" in s and "having" in s):
        return "hard"
    if joins == 1 or agg or "group by" in s:
        return "medium"
    return "easy"


def runs(sql: str, db: str) -> bool:
    """Throwaway check that the gold query executes."""
    try:
        conn = sqlite3.connect(f"file:{DATA}/{db}/{db}.sqlite?mode=ro", uri=True)
        try:
            conn.execute(sql).fetchall()
            return True
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def main() -> None:
    by_db: dict[str, list] = defaultdict(list)
    for ex in json.loads(DEV.read_text()):
        if ex["db_id"] not in EXCLUDE:
            by_db[ex["db_id"]].append(ex)

    rng = random.Random(SEED)
    cases, dropped = [], 0
    for db in sorted(by_db):
        for ex in rng.sample(by_db[db], min(PER_DB, len(by_db[db]))):
            if not runs(ex["query"], db):
                dropped += 1
                continue
            cases.append(
                {
                    "question": ex["question"],
                    "gold_sql": ex["query"],
                    "db_name": db,
                    "difficulty": difficulty(ex["query"]),
                    "source": "spider",
                }
            )

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(cases)} cases across {len(by_db)} DBs ({dropped} dropped).")


if __name__ == "__main__":
    main()
