"""End-to-end validation tests, executed once per backend.

Uses native backend frames — narwhals is an implementation detail.
Backend-specific behaviors (LazyFrame depth, ibis BooleanScalar
normalization, element_wise rejection, nw.Expr accumulation) are covered
in tests/narwhals/test_parity.py and tests/narwhals/test_e2e.py.
"""

import pandas as pd
import pytest

from pandera.api.checks import Check
from pandera.decorators import check_input
from pandera.errors import SchemaDefinitionError, SchemaError, SchemaErrors


def test_valid_frame_passes(DataFrameSchema, Column, frame):
    schema = DataFrameSchema({"x": Column(int), "y": Column(int)})
    result = schema.validate(frame)
    assert result is not None


def test_validate_returns_input_type(DataFrameSchema, Column, frame):
    """schema.validate returns the same frame type as the input."""
    schema = DataFrameSchema({"x": Column(int), "y": Column(int)})
    result = schema.validate(frame)
    assert isinstance(result, type(frame))


def test_missing_column_raises(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(int), "z": Column(int)})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": [1, 2, 3]}))


def test_dtype_mismatch_raises(backend, DataFrameSchema, Column):
    """Wrong column dtype raises SchemaError."""
    schema = DataFrameSchema({"x": Column(str)})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": [1, 2, 3]}))


def test_greater_than_passes(DataFrameSchema, Column, frame):
    schema = DataFrameSchema({"x": Column(int, [Check.greater_than(0)])})
    assert schema.validate(frame) is not None


def test_greater_than_fails(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(int, [Check.greater_than(0)])})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": [-1, 2, -3]}))


def test_isin_passes(DataFrameSchema, Column, frame):
    schema = DataFrameSchema({"x": Column(int, [Check.isin([1, 2, 3])])})
    assert schema.validate(frame) is not None


def test_isin_fails(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(int, [Check.isin([1, 2])])})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": [1, 2, 99]}))


def test_nullable_passes(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(str, nullable=True)})
    assert (
        schema.validate(backend.make_frame({"x": ["a", None, "c"]}))
        is not None
    )


def test_nullable_fails(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(str, nullable=False)})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": ["a", None, "c"]}))


def test_unique_raises(backend, DataFrameSchema, Column):
    """unique=True raises SchemaError on duplicate values."""
    if backend.name == "ibis":
        pytest.xfail(
            "ibis backend does not enforce column-level unique constraint"
        )
    schema = DataFrameSchema({"x": Column(int, unique=True)})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": [1, 1, 3]}))


def test_lazy_mode_collects_all_errors(backend, DataFrameSchema, Column):
    schema = DataFrameSchema(
        {
            "x": Column(int, [Check.greater_than(10)]),
            "y": Column(int, [Check.greater_than(10)]),
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(
            backend.make_frame({"x": [1, 2], "y": [3, 4]}), lazy=True
        )
    assert len(exc_info.value.schema_errors) > 1


def test_custom_bool_check_passes(DataFrameSchema, Column, frame):
    schema = DataFrameSchema({"x": Column(int, [Check(lambda *_: True)])})
    assert schema.validate(frame) is not None


def test_custom_bool_check_fails(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(int, [Check(lambda *_: False)])})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": [1, 2, 3]}))


def test_strict_true_rejects_extra_columns(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"a": Column(int)}, strict=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(backend.make_frame({"a": [1], "b": [2]}))


def test_strict_filter_drops_extra_columns(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"a": Column(int)}, strict="filter")
    result = schema.validate(backend.make_frame({"a": [1], "b": [2]}))
    assert "b" not in result.columns
    assert "a" in result.columns


def test_failure_cases_is_native_type(backend, DataFrameSchema, Column):
    if backend.name == "ibis":
        pytest.xfail(
            "ibis backend returns failure_cases as a pandas DataFrame, not an ibis.Table"
        )
    frame = backend.make_frame({"a": [1, 2, 3]})
    schema = DataFrameSchema({"a": Column(int, [Check.greater_than(10)])})
    with pytest.raises(SchemaError) as exc_info:
        schema.validate(frame)
    assert isinstance(exc_info.value.failure_cases, type(frame))


def test_check_input_decorator(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"a": Column(int)})

    @check_input(schema)
    def my_func(frame):
        return frame

    assert my_func(backend.make_frame({"a": [1, 2, 3]})) is not None


# ---------------------------------------------------------------------------
# drop_invalid_rows parity — lazy=True, filters invalid rows across backends
# ---------------------------------------------------------------------------


def test_drop_invalid_rows_filters_invalid(backend, DataFrameSchema, Column):
    """drop_invalid_rows=True, lazy=True silently drops rows failing any check."""
    schema = DataFrameSchema(
        columns={"a": Column(int, Check.ge(0))},
        drop_invalid_rows=True,
    )
    result = backend.collect(
        schema.validate(backend.make_frame({"a": [-1, 0, 1, 2]}), lazy=True)
    )
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        pd.DataFrame({"a": [0, 1, 2]}),
        check_dtype=False,
    )


def test_drop_invalid_rows_lazy_false_raises(backend, DataFrameSchema, Column):
    """drop_invalid_rows=True with lazy=False raises SchemaDefinitionError."""
    if backend.name == "ibis":
        pytest.xfail(
            "ibis backend raises SchemaError rather than SchemaDefinitionError "
            "for drop_invalid_rows=True with lazy=False"
        )
    schema = DataFrameSchema(
        columns={"a": Column(int, Check.ge(0))},
        drop_invalid_rows=True,
    )
    with pytest.raises(SchemaDefinitionError):
        schema.validate(backend.make_frame({"a": [-1, 1, 2]}), lazy=False)


def test_drop_invalid_rows_nullable(backend, DataFrameSchema, Column):
    """drop_invalid_rows with nullable=True: null rows pass, invalid non-null rows drop."""
    schema = DataFrameSchema(
        columns={"a": Column(int, Check.ge(0), nullable=True)},
        drop_invalid_rows=True,
    )
    result = backend.collect(
        schema.validate(backend.make_frame({"a": [None, -1, 0, 1]}), lazy=True)
    )
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        pd.DataFrame({"a": [float("nan"), 0.0, 1.0]}),
        check_dtype=False,
    )


def test_drop_invalid_rows_multiple_checks(backend, DataFrameSchema, Column):
    """drop_invalid_rows drops a row if ANY per-column check fails.

    Data:
      a=-1, b="0"  → dropped (a fails ge(0))
      a=0,  b="x"  → dropped (b fails isin)
      a=0,  b="0"  → kept
      a=1,  b="1"  → kept
    """
    schema = DataFrameSchema(
        columns={
            "a": Column(int, Check.ge(0)),
            "b": Column(str, Check.isin([*"012"])),
        },
        drop_invalid_rows=True,
    )
    result = backend.collect(
        schema.validate(
            backend.make_frame(
                {"a": [-1, 0, 0, 1], "b": ["0", "x", "0", "1"]}
            ),
            lazy=True,
        )
    )
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        pd.DataFrame({"a": [0, 1], "b": ["0", "1"]}),
        check_dtype=False,
    )
