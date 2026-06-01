"""Compare two SQL result sets for execution accuracy — equal up to column
permutation, and up to row order unless the gold query fixes it."""

from itertools import permutations

from agent.executor import ResultSet

ROUND_TO = 4


def order_matters(sql: str) -> bool:
    return "order by" in sql.lower()


def _cell(value: object) -> object:
    """Normalize one cell so trivially-different encodings compare equal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return round(float(value), ROUND_TO)
    if isinstance(value, str):
        return value.strip()
    return value


def _rows(result_set: ResultSet) -> list[list]:
    return [[_cell(row[col]) for col in result_set.columns] for row in result_set.rows]


def _sort_key(row: list) -> list[tuple]:
    """Rank cells by type so None and mixed-type columns sort without TypeError."""
    ranked: list[tuple] = []
    for cell in row:
        if cell is None:
            ranked.append((0, 0))
        elif isinstance(cell, int | float):
            ranked.append((1, cell))
        else:
            ranked.append((2, str(cell)))
    return ranked


def results_match(gold: ResultSet, predicted: ResultSet, *, ordered: bool) -> bool:
    """True if the result sets hold the same data, ignoring column names and
    order, and row order unless `ordered`."""
    if len(gold.columns) != len(predicted.columns):
        return False
    if len(gold.rows) != len(predicted.rows):
        return False

    gold_rows = _rows(gold)
    pred_rows = _rows(predicted)
    gold_cmp = gold_rows if ordered else sorted(gold_rows, key=_sort_key)

    for perm in permutations(range(len(predicted.columns))):
        cand = [[row[i] for i in perm] for row in pred_rows]
        cand = cand if ordered else sorted(cand, key=_sort_key)
        if cand == gold_cmp:
            return True
    return False
