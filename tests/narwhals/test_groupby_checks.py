"""Tests for groupby-style checks in the Narwhals backend.

The narwhals backend re-implements the pandas dict-based ``groupby`` /
``groups`` contract:

- ``Check(fn, groupby="col")`` partitions the frame by ``col`` and passes
  ``fn`` a ``dict[group_key, value]`` where ``value`` is a Narwhals
  ``Series`` (column-level checks) or ``DataFrame`` (frame-level checks).
- ``Check(fn, groupby=["col_a", "col_b"])`` keys the dict by tuples.
- ``groups=...`` filters the dict before invoking ``fn``.
- ``groupby=callable`` is rejected with ``NotImplementedError``.

The tests use the ``make_narwhals_frame`` fixture which produces frames
across polars (eager), polars (lazy), and ibis (sql-lazy) so groupby is
exercised on every supported native backend.
"""

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from pandera.api.checks import Check
from pandera.api.narwhals.types import NarwhalsData
from pandera.backends.narwhals.checks import NarwhalsCheckBackend

# ---------------------------------------------------------------------------
# Direct backend tests (no schema involved)
# ---------------------------------------------------------------------------


def test_groupby_dict_partitions_single_column(make_narwhals_frame):
    """Single-column groupby returns a dict keyed by scalar values."""
    frame = make_narwhals_frame(
        {"v": [10, 12, 1, 2], "g": ["A", "A", "B", "B"]}
    )
    check = Check(lambda d: True, groupby="g")
    backend = NarwhalsCheckBackend(check)
    grouped = backend.groupby(frame)

    assert set(grouped.keys()) == {"A", "B"}
    assert all(isinstance(v, nw.DataFrame) for v in grouped.values())
    assert sorted(grouped["A"]["v"].to_list()) == [10, 12]
    assert sorted(grouped["B"]["v"].to_list()) == [1, 2]


def test_groupby_dict_partitions_multi_column(make_narwhals_frame):
    """Multi-column groupby returns a dict keyed by tuples."""
    frame = make_narwhals_frame(
        {
            "v": [10, 12, 1, 2],
            "g1": ["A", "A", "B", "B"],
            "g2": [True, True, False, False],
        }
    )
    check = Check(lambda d: True, groupby=["g1", "g2"])
    backend = NarwhalsCheckBackend(check)
    grouped = backend.groupby(frame)

    assert set(grouped.keys()) == {("A", True), ("B", False)}
    assert sorted(grouped[("A", True)]["v"].to_list()) == [10, 12]


def test_groupby_callable_raises(make_narwhals_frame):
    """Passing a callable to ``groupby=`` is not supported (yet)."""
    frame = make_narwhals_frame({"v": [1], "g": ["A"]})
    check = Check(lambda d: True, groupby=lambda df: df.group_by("g"))
    backend = NarwhalsCheckBackend(check)
    with pytest.raises(NotImplementedError, match="Callable groupby"):
        backend.groupby(frame)


def test_apply_groupby_column_check_passes(make_narwhals_frame):
    """A column-level groupby check returns a passing CheckResult."""
    frame = make_narwhals_frame(
        {"v": [10, 12, 1, 2], "g": ["A", "A", "B", "B"]}
    )
    check = Check(
        lambda d: float(d["A"].mean()) > float(d["B"].mean()),
        groupby="g",
    )
    backend = NarwhalsCheckBackend(check)
    nw_data = NarwhalsData(frame, "v")
    out = backend.apply(nw_data)
    assert out is True


def test_apply_groupby_column_check_fails(make_narwhals_frame):
    """A column-level groupby check returns a failing bool."""
    frame = make_narwhals_frame(
        {"v": [1, 2, 10, 12], "g": ["A", "A", "B", "B"]}
    )
    check = Check(
        lambda d: float(d["A"].mean()) > float(d["B"].mean()),
        groupby="g",
    )
    backend = NarwhalsCheckBackend(check)
    nw_data = NarwhalsData(frame, "v")
    assert backend.apply(nw_data) is False


def test_apply_groupby_groups_filter(make_narwhals_frame):
    """``groups=`` filter restricts the dict to the named keys."""
    frame = make_narwhals_frame(
        {"v": [10, 12, 1, 2], "g": ["A", "A", "B", "B"]}
    )
    check = Check(
        lambda d: set(d) == {"A"} and float(d["A"].mean()) > 5,
        groupby="g",
        groups=["A"],
    )
    backend = NarwhalsCheckBackend(check)
    assert backend.apply(NarwhalsData(frame, "v")) is True


def test_apply_groupby_groups_invalid_key_raises(make_narwhals_frame):
    """Requesting a non-existent group surfaces a clear KeyError."""
    frame = make_narwhals_frame({"v": [1, 2], "g": ["A", "A"]})
    check = Check(lambda d: True, groupby="g", groups=["Z"])
    backend = NarwhalsCheckBackend(check)
    with pytest.raises(KeyError, match=r"\['Z'\] provided"):
        backend.apply(NarwhalsData(frame, "v"))


def test_apply_groupby_frame_level_check(make_narwhals_frame):
    """Frame-level groupby checks receive ``dict[key, DataFrame]``."""
    frame = make_narwhals_frame(
        {
            "v1": [1, 2, 3, 4],
            "v2": [10, 20, 30, 40],
            "g": ["A", "A", "B", "B"],
        }
    )
    check = Check(
        lambda d: (
            float(d["A"]["v1"].sum()) == 3 and float(d["B"]["v2"].sum()) == 70
        ),
        groupby="g",
    )
    backend = NarwhalsCheckBackend(check)
    # key="*" → dict values are DataFrames, not Series
    assert backend.apply(NarwhalsData(frame, "*")) is True


# ---------------------------------------------------------------------------
# End-to-end tests via DataFrameSchema
# ---------------------------------------------------------------------------


def test_polars_schema_column_groupby_passes():
    import pandera.polars as pa

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                pl.Int64,
                checks=Check(
                    lambda d: float(d["A"].mean()) > float(d["B"].mean()),
                    groupby="g",
                ),
            ),
            "g": pa.Column(pl.String),
        }
    )
    df = pl.DataFrame({"v": [10, 12, 1, 2], "g": ["A", "A", "B", "B"]})
    out = schema.validate(df)
    assert out.shape == (4, 2)


def test_polars_schema_column_groupby_fails():
    import pandera.polars as pa
    from pandera.errors import SchemaError

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                pl.Int64,
                checks=Check(
                    lambda d: float(d["A"].mean()) > float(d["B"].mean()),
                    groupby="g",
                ),
            ),
            "g": pa.Column(pl.String),
        }
    )
    df = pl.DataFrame({"v": [1, 2, 10, 12], "g": ["A", "A", "B", "B"]})
    with pytest.raises(SchemaError):
        schema.validate(df)


def test_polars_schema_lazy_groupby_passes():
    """Polars LazyFrame is materialized internally for groupby."""
    import pandera.polars as pa

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                pl.Int64,
                checks=Check(lambda d: float(d["A"].sum()) == 22, groupby="g"),
            ),
            "g": pa.Column(pl.String),
        }
    )
    lf = pl.LazyFrame({"v": [10, 12, 1, 2], "g": ["A", "A", "B", "B"]})
    out = schema.validate(lf)
    # validate keeps the result lazy on polars LazyFrame inputs
    assert isinstance(out, pl.LazyFrame)


def test_polars_schema_dataframe_level_multi_groupby():
    """DataFrame-level checks with multi-column groupby use tuple keys."""
    import pandera.polars as pa

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(pl.Int64),
            "g1": pa.Column(pl.String),
            "g2": pa.Column(pl.Boolean),
        },
        checks=Check(
            lambda d: (
                float(d[("A", True)]["v"].sum()) == 22
                and float(d[("B", False)]["v"].sum()) == 3
            ),
            groupby=["g1", "g2"],
        ),
    )
    df = pl.DataFrame(
        {
            "v": [10, 12, 1, 2],
            "g1": ["A", "A", "B", "B"],
            "g2": [True, True, False, False],
        }
    )
    out = schema.validate(df)
    assert out.shape == (4, 3)


def test_ibis_schema_groupby_collects_then_groups():
    """Ibis tables are collected to pandas before iterating groups."""
    ibis = pytest.importorskip("ibis")
    dt = pytest.importorskip("ibis.expr.datatypes")
    pd = pytest.importorskip("pandas")

    import pandera.ibis as pa

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                dt.int64,
                checks=Check(
                    lambda d: float(d["A"].mean()) > float(d["B"].mean()),
                    groupby="g",
                ),
            ),
            "g": pa.Column(dt.string),
        }
    )
    t = ibis.memtable(
        pd.DataFrame({"v": [10, 12, 1, 2], "g": ["A", "A", "B", "B"]})
    )
    out = schema.validate(t)
    assert out.execute().shape == (4, 2)


def test_groupby_single_column_returns_scalar_keys_not_tuples(
    make_narwhals_frame,
):
    """Mirrors pandas: single-column groupby keys are scalars, not 1-tuples.

    Narwhals' native ``group_by`` always yields tuple keys; the backend
    unpacks 1-tuples to a scalar to match the pandas contract.
    """
    frame = make_narwhals_frame({"v": [1, 2, 3], "g": ["A", "A", "B"]})
    check = Check(lambda d: True, groupby="g")
    backend = NarwhalsCheckBackend(check)
    grouped = backend.groupby(frame)
    for key in grouped:
        assert not isinstance(key, tuple), (
            f"Single-column groupby key should be a scalar, got tuple {key!r}"
        )
