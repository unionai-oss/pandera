"""Unit tests for using schemas with Ibis and function decorators."""

import ibis
import pytest

import pandera.ibis as pa
import pandera.typing.ibis as pa_typing


@pytest.fixture
def data() -> ibis.Table:
    return ibis.memtable({"a": [1, 2, 3]})


@pytest.fixture
def invalid_data(data) -> ibis.Table:
    return data.rename({"b": "a"})


def test_ibis_dataframe_check_io(data, invalid_data):
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
        return x.rename({"b": "a"})

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


def test_ibis_dataframe_check_types(data, invalid_data):
    # pylint: disable=unused-argument

    class Model(pa.DataFrameModel):
        a: int

    @pa.check_types
    def fn_check_input(x: pa_typing.Table[Model]): ...

    @pa.check_types
    def fn_check_output(x) -> pa_typing.Table[Model]:
        return x

    @pa.check_types
    def fn_check_io(
        x: pa_typing.Table[Model],
    ) -> pa_typing.Table[Model]:
        return x

    @pa.check_types
    def fn_check_io_invalid(
        x: pa_typing.Table[Model],
    ) -> pa_typing.Table[Model]:
        return x.rename({"b": "a"})  # type: ignore

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
