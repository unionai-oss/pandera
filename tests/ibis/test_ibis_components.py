"""Unit tests for Ibis components."""

from typing import Optional, Union
from collections.abc import Iterable

import ibis
import ibis.expr.datatypes as dt
import pytest
from ibis.common.exceptions import IbisTypeError

import pandera.ibis as pa
from pandera.backends.base import CoreCheckResult
from pandera.backends.ibis.components import ColumnBackend
from pandera.dtypes import DataType
from pandera.engines import ibis_engine
from pandera.errors import SchemaDefinitionError, SchemaError

DTYPES_AND_DATA = [
    # python types
    (int, [1, 2, 3]),
    (str, ["foo", "bar", "baz"]),
    (float, [1.0, 2.0, 3.0]),
    (bool, [True, False, True]),
    # Ibis types
    (dt.Int64, [1, 2, 3]),
    (dt.String, ["foo", "bar", "baz"]),
    (dt.Float64, [1.0, 2.0, 3.0]),
    (dt.Boolean, [True, False, True]),
]


@pytest.mark.parametrize("dtype,data", DTYPES_AND_DATA)
def test_column_schema_simple_dtypes(dtype, data):
    schema = pa.Column(dtype, name="column")
    data = ibis.memtable({"column": data})
    validated_data = schema.validate(data).execute()
    assert validated_data.equals(data.execute())


def test_column_schema_inplace():
    schema = pa.Column(name="column")
    data = ibis.memtable({"column": [1, 2, 3]})
    with pytest.warns(
        UserWarning, match="setting inplace=True will have no effect"
    ):
        schema.validate(data, inplace=True)


def test_column_schema_name_none():
    schema = pa.Column()
    data = ibis.memtable({"column": [1, 2, 3]})
    with pytest.raises(
        SchemaDefinitionError,
        match="Column schema must have a name specified",
    ):
        schema.validate(data)


@pytest.mark.parametrize(
    "column_kwargs",
    [
        {"name": r"^col_\d$", "regex": False},
        {"name": r"col_\d", "regex": True},
    ],
)
def test_column_schema_regex(column_kwargs):
    n_cols = 10
    schema = pa.Column(int, **column_kwargs)
    data = ibis.memtable({f"col_{i}": [1, 2, 3] for i in range(n_cols)})
    validated_data = data.pipe(schema.validate).execute()
    assert validated_data.equals(data.execute())

    for i in range(n_cols):
        invalid_data = data.cast({f"col_{i}": str})
        with pytest.raises(SchemaError):
            invalid_data.pipe(schema.validate)


def test_get_column_backend():
    assert isinstance(
        pa.Column.get_backend(ibis.memtable({"column": [1, 2, 3]})),
        ColumnBackend,
    )
    assert isinstance(
        pa.Column.get_backend(check_type=ibis.Table), ColumnBackend
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": r"^col_\d+$"},
        {"name": r"col_\d+", "regex": True},
    ],
)
def test_get_regex_columns(kwargs):
    column_schema = pa.Column(**kwargs)
    backend = ColumnBackend()
    data = ibis.memtable({f"col_{i}": [1, 2, 3] for i in range(10)})
    matched_columns = backend.get_regex_columns(column_schema, data)
    assert matched_columns == data.columns

    no_match_data = data.rename(
        lambda c: c.replace(
            "col_",
            "foo_",
        )
    )
    with pytest.raises(
        IbisTypeError,
        match="You must select at least one column for a valid projection",
    ):
        matched_columns = backend.get_regex_columns(
            column_schema, no_match_data
        )


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize(
    "data,from_dtype,to_dtype,exception_cls",
    [
        ([1, 2, 3], dt.Int64, dt.String, None),
        ([1, 2, 3], dt.Int64, dt.Float64, None),
        ([0, 1, 0], dt.Int64, dt.Boolean, None),
        ([*"123"], dt.String, dt.Int64, None),
        ([*"123"], dt.String, dt.Float64, None),
        ([*"101"], dt.String, dt.Boolean, SchemaError),
        ([*"abc"], dt.String, dt.Int64, SchemaError),
        ([1.0, 2.0, 3.0], dt.Float64, dt.String, None),
        ([1.0, 2.0, 3.0], dt.Float64, dt.Int64, None),
        ([1.0, 0.0, 1.0], dt.Float64, dt.Boolean, None),
        ([True, False], dt.Boolean, dt.Int64, None),
        ([True, False], dt.Boolean, dt.Float64, None),
        ([True, False], dt.Boolean, dt.String, None),
    ],
)
def test_coerce_dtype(data, from_dtype, to_dtype, exception_cls):
    data = ibis.memtable({"column": data}, schema={"column": from_dtype})
    column_schema = pa.Column(to_dtype, name="column", coerce=True)
    backend = ColumnBackend()

    if exception_cls is None:
        coerced_data = backend.coerce_dtype(data, column_schema)
        assert coerced_data.execute().schema["column"] == to_dtype
    else:
        with pytest.raises(exception_cls):
            backend.coerce_dtype(data, column_schema)


NULLABLE_DTYPES_AND_DATA = [
    [dt.Int64, [1, 2, 3, None]],
    [dt.String, ["foo", "bar", "baz", None]],
    [dt.Float64, [1.0, 2.0, 3.0, float("nan"), None]],
    [dt.Boolean, [True, False, True, None]],
]


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize("dtype, data", NULLABLE_DTYPES_AND_DATA)
@pytest.mark.parametrize("nullable", [True, False])
def test_check_nullable(dtype, data, nullable):
    data = ibis.memtable({"column": data}, schema={"column": dtype})
    column_schema = pa.Column(dtype, nullable=nullable, name="column")
    backend = ColumnBackend()
    check_results: list[CoreCheckResult] = backend.check_nullable(
        data, column_schema
    )
    for result in check_results:
        assert result.passed if nullable else not result.passed


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize("dtype, data", NULLABLE_DTYPES_AND_DATA)
@pytest.mark.parametrize("nullable", [True, False])
def test_check_nullable_regex(dtype, data, nullable):
    data = ibis.memtable(
        {f"column_{i}": data for i in range(3)},
        schema={f"column_{i}": dtype for i in range(3)},
    )
    column_schema = pa.Column(dtype, nullable=nullable, name=r"^column_\d+$")
    backend = ColumnBackend()
    check_results = backend.check_nullable(data, column_schema)
    for result in check_results:
        assert result.passed if nullable else not result.passed


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize("unique", [True, False])
def test_check_unique(unique):
    data = ibis.memtable({"column": [2, 2, 2]})
    column_schema = pa.Column(name="column", unique=unique)
    backend = ColumnBackend()
    check_results = backend.check_unique(data, column_schema)
    for result in check_results:
        assert not result.passed if unique else result.passed


@pytest.mark.parametrize(
    "data,from_dtype",
    [
        ([1, 2, 3], dt.Int64),
        ([*"abc"], dt.String),
        ([1.0, 2.0, 3.0], dt.Float64),
        ([True, False], dt.Boolean),
    ],
)
@pytest.mark.parametrize(
    "check_dtype", [dt.Int64, dt.String, dt.Float64, dt.Boolean]
)
def test_check_dtype(data, from_dtype, check_dtype):
    data = ibis.memtable({"column": data}, schema={"column": from_dtype})
    column_schema = pa.Column(check_dtype, name="column", coerce=True)
    backend = ColumnBackend()

    result = backend.check_dtype(data["column"], column_schema)
    assert result.passed if from_dtype == check_dtype else not result.passed


def test_check_data_container():
    @ibis_engine.Engine.register_dtype
    class MyTestStartsWithID(ibis_engine.String):
        """
        Test DataType which expects strings starting with "id_"
        """

        def check(
            self,
            pandera_dtype: DataType,
            data_container: Optional[  # type:ignore
                # test case doesn't need to be Liskov substitutable
                ibis.Column
            ] = None,
        ) -> Union[bool, Iterable[bool]]:
            return (
                data_container.count(where=data_container.startswith("id_"))  # type: ignore[union-attr]
                .to_pyarrow()
                .as_py()
                == data_container.count().to_pyarrow().as_py()  # type: ignore[union-attr]
            )

        def __str__(self) -> str:
            return str(self.__class__.__name__)

        def __repr__(self) -> str:
            return f"DataType({self})"

    schema = pa.DataFrameSchema(columns={"id": pa.Column(MyTestStartsWithID)})

    data = ibis.memtable({"id": ["id_1", "id_2", "id_3"]})
    schema.validate(data)

    data = ibis.memtable({"id": ["1", "id_2", "id_3"]})
    with pytest.raises(SchemaError):
        schema.validate(data)


@pytest.mark.xfail(reason="not yet implemented")
@pytest.mark.parametrize(
    "data,dtype,default",
    [
        ([1, 2, None], dt.Int64, 3),
        (["a", "b", "c", None], dt.String, "d"),
        ([1.0, 2.0, 3.0, float("nan")], dt.Float64, 4.0),
        ([False, False, False, None], dt.Boolean, True),
    ],
)
def test_set_default(data, dtype, default):
    data = ibis.memtable({"column": data}, schema={"column": dtype})
    column_schema = pa.Column(dtype, name="column", default=default)
    backend = ColumnBackend()
    validated_data = backend.set_default(  # pylint: disable=no-member
        data, column_schema
    ).to_polars()
    assert validated_data.select(
        validated_data["column"].eq(default).any()
    ).item()


@pytest.mark.xfail(raises=NotImplementedError)
def test_expr_as_default():
    schema = pa.DataFrameSchema(
        columns={
            "a": pa.Column(int),
            "b": pa.Column(float, default=1),
            "c": pa.Column(str, default=ibis.literal("foo")),
            "d": pa.Column(int, nullable=True, default="a"),
        },
        add_missing_columns=True,
        coerce=True,
    )
    t = ibis.memtable({"a": [1, 2, 3]})
    assert schema.validate(t).to_polars().to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [1.0, 1.0, 1.0],
        "c": ["foo", "foo", "foo"],
        "d": [1, 2, 3],
    }


@pytest.mark.xfail(raises=NotImplementedError)
def test_missing_with_extra_columns():
    schema = pa.DataFrameSchema(
        columns={
            "a": pa.Column(int),
            "b": pa.Column(float, default=1),
        },
        add_missing_columns=True,
        coerce=True,
    )
    t = ibis.memtable({"a": [1, 2, 3], "c": [4, 5, 6]})
    assert schema.validate(t).collect().to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [1.0, 1.0, 1.0],
        "c": [4, 5, 6],
    }
