"""Verify that built-in checks compose without ``.filter`` chaining.

Stage 5 of ``specs/optimized-strategies.md``: every built-in check
that has a constraint adapter participates in the merged
``FieldConstraints`` and lowers to a single hypothesis strategy
call. This module asserts the structural property by introspecting
the strategy graph after compilation.

A "filter node" is a ``hypothesis.strategies._internal.lazy.LazyStrategy``
or ``FilteredStrategy`` instance. The exact representation can vary
between hypothesis versions, so we count any node whose repr starts
with ``filter`` after stripping wrapper layers.
"""

import hypothesis
import hypothesis.strategies as st

# Trigger constraint adapter registration on the pandas backend.
import pandera.backends.pandas.builtin_checks  # noqa: F401
import pandera.pandas as pa
import pandera.strategies.pandas_strategies as pds
from pandera.api.checks import Check


def _column_elements(schema, name):
    """Return the per-element strategy for a column in ``schema``.

    We construct the field strategy directly so we can inspect its
    structure without going through the dataframe wrapper's
    ``pdst.column``.
    """
    col = schema.columns[name]
    return pds.field_element_strategy(col.dtype, checks=col.checks)


def _count_filter_nodes(strategy) -> int:
    """Count ``.filter(...)`` nodes in the strategy graph."""
    return repr(strategy).count(".filter(")


# ----- numeric --------------------------------------------------------


def test_aggregated_numeric_bounds_emit_zero_filters():
    """``Check.gt(0) & Check.lt(100)`` lowers to one ``from_dtype``."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, checks=[Check.gt(0), Check.lt(100)])}
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) == 0


def test_aggregated_numeric_bounds_with_notin_one_filter():
    """``notin`` is the only allowed trailing filter for numeric stacks."""
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                float,
                checks=[
                    Check.gt(0),
                    Check.lt(100),
                    Check.notin([42.0]),
                ],
            )
        }
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) == 1


def test_redundant_bounds_collapse_to_single_strategy():
    """Multiple ``gt`` checks merge into one bound, not a chain."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, checks=[Check.gt(0), Check.gt(5), Check.gt(10)])}
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) == 0


@hypothesis.given(st.data())
def test_redundant_bounds_emit_only_tightest(data):
    """Three increasing ``gt`` bounds all draw values > the strictest."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, checks=[Check.gt(0), Check.gt(5), Check.gt(10)])}
    )
    df = data.draw(schema.strategy(size=20))
    schema(df)
    assert (df["x"] > 10).all()


# ----- membership ----------------------------------------------------


def test_isin_intersected_with_bounds_emits_zero_filters():
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                checks=[
                    Check.isin(range(0, 100)),
                    Check.gt(10),
                    Check.lt(50),
                ],
            )
        }
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) == 0


def test_isin_with_notin_emits_zero_filters():
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                checks=[
                    Check.isin([1, 2, 3, 4, 5]),
                    Check.notin([3]),
                ],
            )
        }
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) == 0


# ----- strings -------------------------------------------------------


def test_str_startswith_endswith_compose_without_chain():
    """Two regex constraints lower to a single ``st.from_regex``.

    A length filter is allowed (one trailing filter at most).
    """
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                str,
                checks=[
                    Check.str_startswith("foo"),
                    Check.str_endswith("bar"),
                ],
            )
        }
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) == 0


def test_str_startswith_endswith_length_at_most_one_filter():
    """Spec §9.2: regex portion has zero filters; length is at most one."""
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                str,
                checks=[
                    Check.str_startswith("foo"),
                    Check.str_endswith("bar"),
                    Check.str_length(min_value=10),
                ],
            )
        }
    )
    elements = _column_elements(schema, "x")
    assert _count_filter_nodes(elements) <= 1


@hypothesis.given(st.data())
def test_str_combined_constraints_validate(data):
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                str,
                checks=[
                    Check.str_startswith("foo"),
                    Check.str_endswith("bar"),
                    Check.str_length(min_value=10, max_value=20),
                ],
            )
        }
    )
    df = data.draw(schema.strategy(size=10))
    schema(df)
    for s in df["x"]:
        assert s.startswith("foo")
        assert s.endswith("bar")
        assert 10 <= len(s) <= 20


# ----- end-to-end aggregation --------------------------------------


@hypothesis.given(st.data())
def test_aggregated_numeric_with_notin_validates(data):
    """The docs example: gt + lt + notin generates valid data."""
    schema = pa.DataFrameSchema(
        {
            "col": pa.Column(
                int,
                checks=[
                    Check.gt(0),
                    Check.lt(1_000),
                    Check.notin([100, 500]),
                ],
            )
        }
    )
    df = data.draw(schema.strategy(size=20))
    schema(df)


@hypothesis.given(st.data())
def test_isin_intersected_membership_validates(data):
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                checks=[
                    Check.isin(range(0, 100)),
                    Check.gt(10),
                    Check.lt(50),
                    Check.notin([20, 30, 40]),
                ],
            )
        }
    )
    df = data.draw(schema.strategy(size=20))
    schema(df)
    for v in df["x"]:
        assert 10 < v < 50
        assert v not in {20, 30, 40}
