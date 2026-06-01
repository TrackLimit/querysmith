from agent.executor import ResultSet
from eval.compare import order_matters, results_match


def make_result_set(columns, rows):
    return ResultSet(
        columns=columns,
        rows=[dict(zip(columns, r, strict=True)) for r in rows],
    )


def test_row_order_ignored_unless_ordered():
    a = make_result_set(["n"], [(1,), (2,), (3,)])
    b = make_result_set(["n"], [(3,), (1,), (2,)])
    assert results_match(a, b, ordered=False)
    assert not results_match(a, b, ordered=True)


def test_column_order_and_names_ignored():
    gold = make_result_set(["name", "capacity"], [("Wembley", 90000)])
    pred = make_result_set(["cap", "nm"], [(90000, "Wembley")])
    assert results_match(gold, pred, ordered=False)


def test_int_float_equal_and_rounded():
    gold = make_result_set(["v"], [(1,), (1.234511,)])
    pred = make_result_set(["v"], [(1.0,), (1.234519,)])
    assert results_match(gold, pred, ordered=False)


def test_none_sorts_and_matches():
    gold = make_result_set(["v"], [(1,), (None,), (3,)])
    pred = make_result_set(["v"], [(None,), (3,), (1,)])
    assert results_match(gold, pred, ordered=False)


def test_different_values_do_not_match():
    gold = make_result_set(["v"], [(1,), (2,)])
    pred = make_result_set(["v"], [(1,), (9,)])
    assert not results_match(gold, pred, ordered=False)


def test_different_shape_does_not_match():
    assert not results_match(
        make_result_set(["a"], [(1,)]),
        make_result_set(["a", "b"], [(1, 2)]),
        ordered=False,
    )


def test_order_matters_reads_the_gold_sql():
    assert order_matters("SELECT name FROM t ORDER BY age")
    assert not order_matters("SELECT name FROM t WHERE age > 5")
