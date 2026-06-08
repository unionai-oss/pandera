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
