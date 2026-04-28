"""End-to-end validation tests, executed once per backend.

Uses native backend frames — narwhals is an implementation detail.
Backend-specific behaviors (LazyFrame depth, ibis BooleanScalar normalization,
etc.) are covered in tests/narwhals/test_e2e.py.
"""

import pytest

from pandera.api.checks import Check
from pandera.decorators import check_input
from pandera.errors import SchemaError, SchemaErrors


def test_valid_frame_passes(DataFrameSchema, Column, frame):
    schema = DataFrameSchema({"x": Column(int), "y": Column(int)})
    result = schema.validate(frame)
    assert result is not None


def test_missing_column_raises(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(int), "z": Column(int)})
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
    assert schema.validate(backend.make_frame({"x": ["a", None, "c"]})) is not None


def test_nullable_fails(backend, DataFrameSchema, Column):
    schema = DataFrameSchema({"x": Column(str, nullable=False)})
    with pytest.raises(SchemaError):
        schema.validate(backend.make_frame({"x": ["a", None, "c"]}))


def test_lazy_mode_collects_all_errors(backend, DataFrameSchema, Column):
    schema = DataFrameSchema(
        {
            "x": Column(int, [Check.greater_than(10)]),
            "y": Column(int, [Check.greater_than(10)]),
        }
    )
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(backend.make_frame({"x": [1, 2], "y": [3, 4]}), lazy=True)
    assert len(exc_info.value.schema_errors) > 1


def test_custom_bool_check_passes(DataFrameSchema, Column, frame):
    schema = DataFrameSchema(
        {"x": Column(int, [Check(lambda native_frame, key: True)])}
    )
    assert schema.validate(frame) is not None


def test_custom_bool_check_fails(backend, DataFrameSchema, Column):
    schema = DataFrameSchema(
        {"x": Column(int, [Check(lambda native_frame, key: False)])}
    )
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
