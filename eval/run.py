"""Run the eval set: NL-to-SQL, execute against gold, compare. Reports the score
and fails below the floor."""

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from agent.executor import Error, execute_sql
from agent.ingest import ingest_schema
from agent.loop import solve_sql
from eval.compare import order_matters, results_match

CASE_FILES = [Path("eval/spider_cases.jsonl"), Path("eval/adversarial_cases.jsonl")]
FLOOR = 0.70
DIFFICULTIES = ["easy", "medium", "hard", "extra"]


def _db_path(db_name: str) -> str:
    return f"sqlite-data/{db_name}/{db_name}.sqlite"


def load_cases() -> list[dict]:
    cases = []
    for path in CASE_FILES:
        for line in path.read_text().splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return cases


def evaluate(case: dict, chroma_path: str) -> dict:
    path = _db_path(case["db_name"])
    gold = execute_sql(case["gold_sql"], path)
    if isinstance(gold, Error):
        print(f"WARNING gold failed [{case['db_name']}]: {gold.message}")
        return {**case, "generated_sql": None, "valid": False, "passed": False}

    generated = solve_sql(case["question"], path, chroma_path=chroma_path)
    if generated is None:
        return {**case, "generated_sql": None, "valid": False, "passed": False}

    predicted = execute_sql(generated, path)
    if isinstance(predicted, Error):
        return {**case, "generated_sql": generated, "valid": False, "passed": False}

    passed = results_match(gold, predicted, ordered=order_matters(case["gold_sql"]))
    return {**case, "generated_sql": generated, "valid": True, "passed": passed}


def run() -> list[dict]:
    by_db: dict[str, list[dict]] = defaultdict(list)
    for case in load_cases():
        by_db[case["db_name"]].append(case)

    # isolated from the CLI's chroma_db/ so the eval can't corrupt it
    chroma_path = tempfile.mkdtemp(prefix="chroma_eval_")
    results = []
    for db_name in sorted(by_db):
        ingest_schema(_db_path(db_name), chroma_path=chroma_path)
        print(f"[{db_name}] {len(by_db[db_name])} cases")
        results.extend(evaluate(case, chroma_path) for case in by_db[db_name])
    return results


def report(results: list[dict]) -> None:
    for r in results:
        if not r["passed"]:
            print(f"\nFAIL [{r['db_name']}/{r['difficulty']}] {r['question']}")
            print(f"  gold: {r['gold_sql']}")
            print(f"  pred: {r['generated_sql']}")

    total = len(results)
    valid = sum(r["valid"] for r in results)
    passed = sum(r["passed"] for r in results)
    print(f"\nexecution accuracy: {passed / total:.1%} ({passed}/{total})")
    print(f"valid-SQL rate:     {valid / total:.1%} ({valid}/{total})")


def markdown_report(results: list[dict]) -> str:
    lines = [
        "# Execution-accuracy report",
        "",
        "| difficulty | passed | total | accuracy |",
        "|---|---|---|---|",
    ]
    for diff in DIFFICULTIES:
        bucket = [r for r in results if r["difficulty"] == diff]
        if bucket:
            p = sum(r["passed"] for r in bucket)
            lines.append(f"| {diff} | {p} | {len(bucket)} | {p / len(bucket):.0%} |")
    total = len(results)
    passed = sum(r["passed"] for r in results)
    valid = sum(r["valid"] for r in results)
    lines += [
        f"| **all** | **{passed}** | **{total}** | **{passed / total:.1%}** |",
        "",
        f"valid-SQL rate: {valid}/{total} ({valid / total:.1%})",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    results = run()
    report(results)
    Path("eval/report.md").write_text(markdown_report(results))
    accuracy = sum(r["passed"] for r in results) / len(results)
    if accuracy < FLOOR:
        print(f"\naccuracy {accuracy:.1%} below floor {FLOOR:.0%} — failing")
        sys.exit(1)
