"""Unit tests for using schemas with polars and function decorators."""

import polars as pl
import pytest

import pandera.polars as pa
import pandera.typing.polars as pa_typing


@pytest.fixture
def data() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3]})


@pytest.fixture
def invalid_data(data) -> pl.DataFrame:
    return data.rename({"a": "b"})


def test_polars_dataframe_check_io(data, invalid_data):
    # pylint: disable=unused-argument

    schema = pa.DataFrameSchema({"a": pa.Column(int)})

    @pa.check_input(schema)
    def fn_check_input(x): ...

    @pa.check_output(schema)
    def fn_check_output(x):
        return x

    @pa.check_io(x=schema, out=schema)
    def fn_check_io(x):
        return x

    @pa.check_io(x=schema, out=schema)
    def fn_check_io_invalid(x):
        return x.rename({"a": "b"})

    # valid data should pass
    fn_check_input(data)
    fn_check_output(data)
    fn_check_io(data)

    # invalid data or invalid function should not pass
    with pytest.raises(pa.errors.SchemaError):
        fn_check_input(invalid_data)

    with pytest.raises(pa.errors.SchemaError):
        fn_check_output(invalid_data)

    with pytest.raises(pa.errors.SchemaError):
        fn_check_io_invalid(data)


def test_polars_dataframe_check_types(data, invalid_data):
    # pylint: disable=unused-argument

    class Model(pa.DataFrameModel):
        a: int

    @pa.check_types
    def fn_check_input(x: pa_typing.DataFrame[Model]): ...

    @pa.check_types
    def fn_check_output(x) -> pa_typing.DataFrame[Model]:
        return x

    @pa.check_types
    def fn_check_io(
        x: pa_typing.DataFrame[Model],
    ) -> pa_typing.DataFrame[Model]:
        return x

    @pa.check_types
    def fn_check_io_invalid(
        x: pa_typing.DataFrame[Model],
    ) -> pa_typing.DataFrame[Model]:
        return x.rename({"a": "b"})  # type: ignore

    # valid data should pass
    fn_check_input(data)
    fn_check_output(data)
    fn_check_io(data)

    # invalid data or invalid function should not pass
    with pytest.raises(pa.errors.SchemaError):
        fn_check_input(invalid_data)

    with pytest.raises(pa.errors.SchemaError):
        fn_check_output(invalid_data)

    with pytest.raises(pa.errors.SchemaError):
        fn_check_io_invalid(data)
