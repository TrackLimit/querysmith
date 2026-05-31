"""Run the eval set: NL-to-SQL, execute against gold, compare. Prints the score."""

import json
import tempfile
from collections import defaultdict
from pathlib import Path

from agent.cli import generate_sql
from agent.executor import Error, execute_sql
from agent.ingest import ingest_schema
from eval.compare import order_matters, results_match

CASE_FILES = [Path("eval/spider_cases.jsonl"), Path("eval/adversarial_cases.jsonl")]


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

    generated = generate_sql(case["question"], path, chroma_path=chroma_path)
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


if __name__ == "__main__":
    report(run())
