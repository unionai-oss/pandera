"""Tests for the validation decorators on DataFusion DataFrames."""

import datafusion
import pytest

import pandera.datafusion as pa
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing.datafusion import DataFrame


class Model(pa.DataFrameModel):
    a: int = pa.Field(gt=0)
    b: str


VALID = {"a": [1], "b": ["x"]}


@pytest.fixture
def schema():
    return Model.to_schema()


def test_check_types(make_df):
    @pa.check_types
    def transform(df: DataFrame[Model]) -> DataFrame[Model]:
        return df

    validated = transform(make_df(VALID))
    assert isinstance(validated, datafusion.DataFrame)
    assert validated.to_pydict() == VALID

    with pytest.raises(SchemaError):
        transform(make_df({"a": ["x"], "b": ["x"]}))
    with pytest.raises(SchemaError):
        transform(make_df({"a": [-1], "b": ["x"]}))


def test_check_types_validates_output(make_df):
    @pa.check_types
    def transform(df: DataFrame[Model]) -> DataFrame[Model]:
        return df.drop("b")

    with pytest.raises(SchemaError):
        transform(make_df(VALID))


def test_check_types_lazy(make_df):
    @pa.check_types(lazy=True)
    def transform(df: DataFrame[Model]) -> DataFrame[Model]:
        return df

    with pytest.raises(SchemaErrors) as exc_info:
        transform(make_df({"a": [-1], "b": [1]}))
    assert len(exc_info.value.schema_errors) == 2


def test_check_types_ignores_unannotated_arguments(make_df):
    @pa.check_types
    def transform(df: DataFrame[Model], factor: int) -> DataFrame[Model]:
        return df

    assert transform(make_df(VALID), 2).to_pydict() == VALID


def test_check_types_data_format_conversion():
    class In(pa.DataFrameModel):
        a: int = pa.Field(gt=0)

        class Config:
            from_format = "dict"

    class Out(In):
        class Config:
            from_format = None
            to_format = "dict"

    @pa.check_types
    def transform(df: DataFrame[In]) -> DataFrame[Out]:
        assert isinstance(df, datafusion.DataFrame)
        return df

    assert transform({"a": [1, 2]}) == {"a": [1, 2]}
    with pytest.raises(SchemaError):
        transform({"a": [-1]})


def test_check_input(make_df, schema):
    @pa.check_input(schema)
    def transform(df):
        return df

    assert transform(make_df(VALID)).to_pydict() == VALID
    with pytest.raises(SchemaError, match="check_input"):
        transform(make_df({"a": [-1], "b": ["x"]}))


def test_check_output(make_df, schema):
    @pa.check_output(schema)
    def transform(df):
        return df

    assert transform(make_df(VALID)).to_pydict() == VALID
    with pytest.raises(SchemaError, match="check_output"):
        transform(make_df({"a": [-1], "b": ["x"]}))


def test_check_io(make_df, schema):
    @pa.check_io(df=schema, out=schema)
    def transform(df):
        return df

    @pa.check_io(out=schema)
    def breaks_output(df):
        return df.drop("b")

    assert transform(make_df(VALID)).to_pydict() == VALID
    with pytest.raises(SchemaError):
        transform(make_df({"a": [-1], "b": ["x"]}))
    with pytest.raises(SchemaError):
        breaks_output(make_df(VALID))
