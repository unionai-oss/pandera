"""Unit tests for PydanticModel dtype with polars backend."""

import polars as pl
import pytest
from pydantic import BaseModel

import pandera.polars as pa
from pandera.engines.polars_engine import PydanticModel


class Record(BaseModel):
    """Pydantic record model."""

    name: str
    xcoord: int
    ycoord: int


class PydanticSchema(pa.DataFrameModel):
    """Pandera schema using the pydantic model."""

    class Config:
        """Config with dataframe-level data type."""

        dtype = PydanticModel(Record)


def test_pydantic_model():
    """Test that pydantic model correctly validates polars data."""
    valid_df = pl.DataFrame(
        {
            "name": ["foo", "bar", "baz"],
            "xcoord": [1, 2, 3],
            "ycoord": [4, 5, 6],
        }
    )

    invalid_df = pl.DataFrame(
        {
            "name": ["foo", "bar", "baz"],
            "xcoord": ["1", "2", "c"],
            "ycoord": ["4", "5", "d"],
        }
    )

    validated = PydanticSchema.validate(valid_df)
    assert isinstance(validated, pl.DataFrame)
    assert validated.shape == valid_df.shape

    with pytest.raises(pa.errors.SchemaError):
        PydanticSchema.validate(invalid_df)


def test_pydantic_model_coercion():
    """Test that pydantic model coerces types correctly."""
    df = pl.DataFrame(
        {
            "name": ["foo", "bar", "baz"],
            "xcoord": [1.0, 2.0, 3.0],
            "ycoord": [4.0, 5.0, 6.0],
        }
    )

    validated = PydanticSchema.validate(df)
    assert isinstance(validated, pl.DataFrame)
    assert validated["xcoord"].to_list() == [1, 2, 3]
    assert validated["ycoord"].to_list() == [4, 5, 6]


def test_pydantic_model_with_lazyframe():
    """Test that pydantic model works with LazyFrame input."""
    lf = pl.LazyFrame(
        {
            "name": ["foo", "bar"],
            "xcoord": [1, 2],
            "ycoord": [3, 4],
        }
    )

    validated = PydanticSchema.validate(lf)
    assert isinstance(validated, pl.LazyFrame)
    result = validated.collect()
    assert result.shape == (2, 3)


def test_pydantic_model_missing_columns():
    """Test that pydantic model raises on missing columns."""
    df = pl.DataFrame(
        {
            "name": ["foo", "bar"],
            "xcoord": [1, 2],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        PydanticSchema.validate(df)


def test_pydantic_model_empty_dataframe():
    """Test that pydantic model handles empty dataframe with warning."""
    df = pl.DataFrame(
        {
            "name": pl.Series([], dtype=pl.Utf8),
            "xcoord": pl.Series([], dtype=pl.Int64),
            "ycoord": pl.Series([], dtype=pl.Int64),
        }
    )

    with pytest.warns(UserWarning, match="PydanticModel cannot validate"):
        validated = PydanticSchema.validate(df)
    assert isinstance(validated, pl.DataFrame)
    assert validated.shape == (0, 3)


def test_pydantic_model_empty_dataframe_missing_column():
    """Test that pydantic model raises on empty df with missing columns."""
    df = pl.DataFrame(
        {
            "name": pl.Series([], dtype=pl.Utf8),
            "xcoord": pl.Series([], dtype=pl.Int64),
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        PydanticSchema.validate(df)


def test_pydantic_model_empty_via_model():
    """Test that DataFrameModel.empty() works with PydanticModel dtype."""

    class Row(BaseModel):
        name: str | None = None
        value: int | None = None

    class MySchema(pa.DataFrameModel):
        class Config:
            dtype = PydanticModel(Row)
            coerce = True
            strict = False

    result = MySchema.empty()
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 0
    assert result.schema == {"name": pl.String, "value": pl.Int64}


def test_pydantic_model_get_polars_schema_unsupported_type():
    """Test that _get_polars_schema falls back to pl.Null for unsupported types."""

    class Custom:
        pass

    class ModelWithCustom(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        field: Custom

    pm = PydanticModel(ModelWithCustom)
    schema = pm._get_polars_schema()
    assert schema == {"field": pl.Null}


def test_pydantic_model_get_polars_schema_typing_union():
    """typing.Union[X, None] (not PEP 604) should hit the `origin is Union` branch."""
    import typing
    from types import SimpleNamespace

    optional_int = typing.Union[int, None]
    fake_field = SimpleNamespace(outer_type_=optional_int)
    fake_model = type("FakeModel", (), {"__fields__": {"a": fake_field}})

    pm = PydanticModel.__new__(PydanticModel)
    object.__setattr__(pm, "type", fake_model)

    assert pm._get_polars_schema() == {"a": pl.Int64}


def test_pydantic_model_get_polars_schema_multiarg_union():
    """Union with >1 non-None arg can't be unwrapped and falls back to pl.Null."""
    import typing
    from types import SimpleNamespace

    multi_union = typing.Union[int, str, None]
    fake_field = SimpleNamespace(outer_type_=multi_union)
    fake_model = type("FakeModel", (), {"__fields__": {"x": fake_field}})

    pm = PydanticModel.__new__(PydanticModel)
    object.__setattr__(pm, "type", fake_model)

    assert pm._get_polars_schema() == {"x": pl.Null}


def test_pydantic_model_get_polars_schema_v1_fallback():
    """Test _get_polars_schema v1 branch via a mock without model_fields."""
    from types import SimpleNamespace

    fake_field = SimpleNamespace(outer_type_=str | None)
    fake_model = type("FakeModel", (), {"__fields__": {"name": fake_field}})

    pm = PydanticModel.__new__(PydanticModel)
    object.__setattr__(pm, "type", fake_model)

    schema = pm._get_polars_schema()
    assert schema == {"name": pl.String}


@pytest.mark.parametrize("coerce", [True, False])
def test_pydantic_model_coerce_always_true(coerce: bool):
    """Test that DataFrameSchema.coerce is always True with PydanticModel."""
    schema = pa.DataFrameSchema(dtype=PydanticModel(Record), coerce=coerce)
    assert schema.coerce is True


def test_pydantic_model_schema_validate():
    """Test PydanticModel via DataFrameSchema.validate."""
    schema = pa.DataFrameSchema(dtype=PydanticModel(Record))

    valid_df = pl.DataFrame(
        {
            "name": ["foo", "bar"],
            "xcoord": [1, 2],
            "ycoord": [3, 4],
        }
    )

    validated = schema.validate(valid_df)
    assert isinstance(validated, pl.DataFrame)

    invalid_df = pl.DataFrame(
        {
            "name": ["foo", "bar"],
            "xcoord": ["a", "b"],
            "ycoord": ["c", "d"],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        schema.validate(invalid_df)


def test_pydantic_model_nullable_column_late_value():
    """Regression: a nullable string column whose first 100+ rows are None
    must not be inferred as pl.Null, which would drop later real values."""

    class Record(BaseModel):
        name: str | None = None
        code: str | None = None

    class RecordSchema(pa.DataFrameModel):
        class Config:
            dtype = PydanticModel(Record)
            coerce = True
            strict = False

    # First 120 rows have code=None; a later row carries a real string value
    # beyond the default infer_schema_length window of 100.
    rows = [{"name": "A", "code": None} for _ in range(120)]
    rows.append({"name": "B", "code": "00435L108"})
    df = pl.DataFrame(rows, schema={"name": pl.Utf8, "code": pl.Utf8})

    validated = RecordSchema.validate(df)
    assert isinstance(validated, pl.DataFrame)
    assert validated.schema["code"] == pl.String
    assert validated["code"].to_list()[-1] == "00435L108"
    assert validated.shape == (121, 2)


def test_pydantic_model_fully_null_column_keeps_dtype():
    """Regression: a fully-null nullable column infers as pl.Null on its own.
    It should be repaired to a concrete dtype so downstream ops don't break."""

    class Record(BaseModel):
        name: str | None = None
        code: str | None = None

    class RecordSchema(pa.DataFrameModel):
        class Config:
            dtype = PydanticModel(Record)
            coerce = True
            strict = False

    rows = [{"name": "A", "code": None} for _ in range(5)]
    df = pl.DataFrame(rows, schema={"name": pl.Utf8, "code": pl.Utf8})

    validated = RecordSchema.validate(df)
    assert isinstance(validated, pl.DataFrame)
    assert validated.schema["code"] == pl.String
    assert validated["code"].to_list() == [None] * 5


def test_pydantic_model_fully_null_column_uses_coerced_dtype():
    """A fully-null column should be repaired to the pydantic model's coerced
    output dtype, not the (possibly different) input dtype.

    Here the input is Utf8 but the field is `int | None`, so the output column
    should be an integer type, consistent with PydanticModel.empty().
    """

    class Record(BaseModel):
        code: int | None = None

    class RecordSchema(pa.DataFrameModel):
        class Config:
            dtype = PydanticModel(Record)
            coerce = True
            strict = False

    df = pl.DataFrame({"code": pl.Series([None, None], dtype=pl.Utf8)})

    validated = RecordSchema.validate(df)
    assert isinstance(validated, pl.DataFrame)
    assert validated.schema["code"] == pl.Int64
    assert validated["code"].to_list() == [None, None]


def test_pydantic_model_fully_null_column_unrepairable_dtype():
    """When a fully-null column maps to pl.Null in both the model schema and
    the input schema, there is no concrete dtype to repair it with, so it is
    left as pl.Null (exercises the _repair_dtype fall-through)."""

    class Custom:
        pass

    class Record(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        blob: Custom | None = None

    class RecordSchema(pa.DataFrameModel):
        class Config:
            dtype = PydanticModel(Record)
            coerce = True
            strict = False

    # No explicit schema: an all-None column infers as pl.Null on input too,
    # and Custom is unsupported so the model schema is pl.Null as well.
    df = pl.DataFrame({"blob": [None, None]})
    assert df.schema["blob"] == pl.Null

    validated = RecordSchema.validate(df)
    assert isinstance(validated, pl.DataFrame)
    assert validated.schema["blob"] == pl.Null
    assert validated["blob"].to_list() == [None, None]


def test_pydantic_model_nullable_late_value_failure_path():
    """The failure-path frame construction must also scan all rows so the
    ParserError surfaces real parser failure cases rather than the polars
    "could not append value" inference error from a Null-typed builder."""

    class Record(BaseModel):
        name: str | None = None
        code: str | None = None
        xcoord: int

    class RecordSchema(pa.DataFrameModel):
        class Config:
            dtype = PydanticModel(Record)
            coerce = True
            strict = False

    rows = [{"name": "A", "code": None, "xcoord": 1} for _ in range(120)]
    rows.append({"name": "B", "code": "00435L108", "xcoord": 2})
    # One invalid row triggers the failure path (fc_df) construction.
    rows.append({"name": "C", "code": "X", "xcoord": "not-an-int"})
    df = pl.DataFrame(
        rows,
        schema={"name": pl.Utf8, "code": pl.Utf8, "xcoord": pl.Utf8},
    )

    with pytest.raises(pa.errors.SchemaError) as exc_info:
        RecordSchema.validate(df)

    # The error must be the intended parser failure, not the polars
    # "could not append value" inference error that the bug produced.
    assert "could not append value" not in str(exc_info.value)

    # The failure_cases frame must carry the genuinely-invalid row (the bad
    # xcoord), and the late "code" value must have been preserved as a string
    # rather than dropped by a Null-typed builder.
    failure_cases = exc_info.value.failure_cases
    assert failure_cases.schema["code"] == pl.String
    assert failure_cases["xcoord"].to_list() == ["not-an-int"]


def test_pydantic_model_with_check_types():
    """Test PydanticModel with @check_types decorator."""
    from pandera.typing.polars import DataFrame as PolarsDataFrame

    @pa.check_types
    def process(
        df: PolarsDataFrame[PydanticSchema],
    ) -> PolarsDataFrame[PydanticSchema]:
        return df

    valid_df = pl.DataFrame(
        {
            "name": ["foo", "bar"],
            "xcoord": [1, 2],
            "ycoord": [3, 4],
        }
    )

    result = process(valid_df)
    assert isinstance(result, pl.DataFrame)
