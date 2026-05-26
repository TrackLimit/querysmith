"""Measure retrieval quality (recall@k) over a hand-labeled gold set."""

from agent.ingest import ingest_schema
from agent.retrieval import retrieve

DB_PATH = "sqlite-data/concert_singer/concert_singer.sqlite"

GOLD: list[tuple[str, set[str]]] = [
    ("How many singers are there?", {"singer"}),
    ("What is the average age of all singers?", {"singer"}),
    ("Which singers are from France?", {"singer"}),
    ("Name the stadium with the largest capacity.", {"stadium"}),
    ("How many stadiums are there?", {"stadium"}),
    ("List the themes of all concerts.", {"concert"}),
    ("How many concerts were held in 2014?", {"concert"}),
    ("Which stadium hosted each concert?", {"concert", "stadium"}),
    ("How many singers performed at each concert?", {"concert", "singer_in_concert"}),
    ("Which singers never performed at a concert?", {"singer", "singer_in_concert"}),
    (
        "Show each singer with the concerts they performed in.",
        {"singer", "concert", "singer_in_concert"},
    ),
]


def recall_at_k(k: int) -> float:
    found = total = 0
    for question, expected in GOLD:
        retrieved = {t.name for t in retrieve(question, k)}
        found += len(expected & retrieved)
        total += len(expected)
    return found / total


if __name__ == "__main__":
    ingest_schema(DB_PATH)
    for k in (1, 2, 3):
        print(f"recall@{k}: {recall_at_k(k):.2f}")
