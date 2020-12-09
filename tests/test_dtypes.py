"""Tests a variety of python and pandas dtypes, and tests some specific
coercion examples."""

import numpy as np
import pandas as pd
import pytest
from packaging import version

import pandera as pa
from pandera import (
    Bool,
    Category,
    Check,
    Column,
    DataFrameSchema,
    DateTime,
    Float,
    Int,
    Object,
    PandasDtype,
    SeriesSchema,
    String,
    Timedelta,
)
from pandera.dtypes import (
    _DEFAULT_NUMPY_FLOAT_TYPE,
    _DEFAULT_NUMPY_INT_TYPE,
    _DEFAULT_PANDAS_FLOAT_TYPE,
    _DEFAULT_PANDAS_INT_TYPE,
)
from pandera.errors import SchemaError

PANDAS_VERSION = version.parse(pd.__version__)

TESTABLE_DTYPES = [
    (Bool, "bool"),
    (DateTime, "datetime64[ns]"),
    (Category, "category"),
    (Float, Float.str_alias),
    (Int, Int.str_alias),
    (Object, "object"),
    (String, String.str_alias),
    (Timedelta, "timedelta64[ns]"),
    ("bool", "bool"),
    ("datetime64[ns]", "datetime64[ns]"),
    ("category", "category"),
    ("float64", "float64"),
]


def test_default_numeric_dtypes():
    """Test that default numeric dtypes int and float are consistent."""
    assert str(pd.Series([1]).dtype) == _DEFAULT_PANDAS_INT_TYPE
    assert pa.Int.str_alias == _DEFAULT_PANDAS_INT_TYPE
    assert str(pd.Series([1], dtype=int).dtype) == _DEFAULT_NUMPY_INT_TYPE
    assert str(pd.Series([1], dtype="int").dtype) == _DEFAULT_NUMPY_INT_TYPE

    assert str(pd.Series([1.0]).dtype) == _DEFAULT_PANDAS_FLOAT_TYPE
    assert pa.Float.str_alias == _DEFAULT_PANDAS_FLOAT_TYPE
    assert (
        str(pd.Series([1.0], dtype=float).dtype) == _DEFAULT_NUMPY_FLOAT_TYPE
    )
    assert (
        str(pd.Series([1.0], dtype="float").dtype) == _DEFAULT_NUMPY_FLOAT_TYPE
    )


def test_numeric_dtypes():
    """Test every numeric type can be validated properly by schema.validate"""
    for dtype in [pa.Float, pa.Float16, pa.Float32, pa.Float64]:
        assert all(
            isinstance(
                schema.validate(
                    pd.DataFrame(
                        {"col": [-123.1, -7654.321, 1.0, 1.1, 1199.51, 5.1]},
                        dtype=dtype.str_alias,
                    )
                ),
                pd.DataFrame,
            )
            for schema in [
                DataFrameSchema({"col": Column(dtype, nullable=False)}),
                DataFrameSchema(
                    {"col": Column(dtype.str_alias, nullable=False)}
                ),
            ]
        )

    for dtype in [pa.Int, pa.Int8, pa.Int16, pa.Int32, pa.Int64]:
        assert all(
            isinstance(
                schema.validate(
                    pd.DataFrame(
                        {"col": [-712, -4, -321, 0, 1, 777, 5, 123, 9000]},
                        dtype=dtype.str_alias,
                    )
                ),
                pd.DataFrame,
            )
            for schema in [
                DataFrameSchema({"col": Column(dtype, nullable=False)}),
                DataFrameSchema(
                    {"col": Column(dtype.str_alias, nullable=False)}
                ),
            ]
        )

    for dtype in [pa.UInt8, pa.UInt16, pa.UInt32, pa.UInt64]:
        assert all(
            isinstance(
                schema.validate(
                    pd.DataFrame(
                        {"col": [1, 777, 5, 123, 9000]}, dtype=dtype.str_alias
                    )
                ),
                pd.DataFrame,
            )
            for schema in [
                DataFrameSchema({"col": Column(dtype, nullable=False)}),
                DataFrameSchema(
                    {"col": Column(dtype.str_alias, nullable=False)}
                ),
            ]
        )


@pytest.mark.skipif(
    PANDAS_VERSION.release < (1, 0, 0),  # type: ignore
    reason="pandas >= 1.0.0 required",
)
@pytest.mark.parametrize(
    "dtype",
    [
        pa.INT8,
        pa.INT16,
        pa.INT32,
        pa.INT64,
        pa.UINT8,
        pa.UINT16,
        pa.UINT32,
        pa.UINT64,
    ],
)
@pytest.mark.parametrize("coerce", [True, False])
def test_pandas_nullable_int_dtype(dtype, coerce):
    """Test that pandas nullable int dtype can be specified in a schema."""
    assert all(
        isinstance(
            schema.validate(
                pd.DataFrame(
                    # keep max range to 127 in order to support Int8
                    {"col": range(128)},
                    **({} if coerce else {"dtype": dtype.str_alias}),
                )
            ),
            pd.DataFrame,
        )
        for schema in [
            DataFrameSchema(
                {"col": Column(dtype, nullable=False)}, coerce=coerce
            ),
            DataFrameSchema(
                {"col": Column(dtype.str_alias, nullable=False)}, coerce=coerce
            ),
        ]
    )


@pytest.mark.parametrize("str_alias", ["foo", "bar", "baz", "asdf", "qwerty"])
def test_unrecognized_str_aliases(str_alias):
    """Test that unrecognized string aliases are supported."""
    with pytest.raises(TypeError):
        PandasDtype.from_str_alias(str_alias)


def test_category_dtype():
    """Test the category type can be validated properly by schema.validate"""
    schema = DataFrameSchema(
        columns={
            "col": Column(
                pa.Category,
                checks=[
                    Check(lambda s: set(s) == {"A", "B", "C"}),
                    Check(
                        lambda s: s.cat.categories.tolist() == ["A", "B", "C"]
                    ),
                    Check(lambda s: s.isin(["A", "B", "C"])),
                ],
                nullable=False,
            ),
        },
        coerce=False,
    )
    validated_df = schema.validate(
        pd.DataFrame(
            {"col": pd.Series(["A", "B", "A", "B", "C"], dtype="category")}
        )
    )
    assert isinstance(validated_df, pd.DataFrame)


def test_category_dtype_coerce():
    """Test coercion of the category type is validated properly by
    schema.validate and fails safely."""
    columns = {
        "col": Column(
            pa.Category,
            checks=Check(lambda s: set(s) == {"A", "B", "C"}),
            nullable=False,
        ),
    }

    with pytest.raises(SchemaError):
        DataFrameSchema(columns=columns, coerce=False).validate(
            pd.DataFrame(
                {"col": pd.Series(["A", "B", "A", "B", "C"], dtype="object")}
            )
        )

    validated_df = DataFrameSchema(columns=columns, coerce=True).validate(
        pd.DataFrame(
            {"col": pd.Series(["A", "B", "A", "B", "C"], dtype="object")}
        )
    )
    assert isinstance(validated_df, pd.DataFrame)


def helper_type_validation(dataframe_type, schema_type, debugging=False):
    """
    Helper function for using same or different dtypes for the dataframe and
    the schema_type
    """
    df = pd.DataFrame({"column1": [dataframe_type(1)]})
    if debugging:
        print(dataframe_type, df.column1)
    schema = pa.DataFrameSchema({"column1": pa.Column(schema_type)})
    if debugging:
        print(schema)
    schema(df)


def test_numpy_type():
    """Test various numpy dtypes"""
    # Test correct conversions
    valid_types = (
        # Pandas always converts complex numbers to np.complex128
        (np.complex, np.complex),
        (np.complex, np.complex128),
        (np.complex128, np.complex),
        (np.complex64, np.complex128),
        (np.complex128, np.complex128),
        # Pandas always converts float numbers to np.float64
        (np.float, np.float),
        (np.float, np.float64),
        (np.float16, np.float64),
        (np.float32, np.float64),
        (np.float64, np.float64),
        # Pandas always converts int numbers to np.int64
        (np.int, np.int),
        (np.int, np.int64),
        (np.int8, np.int64),
        (np.int16, np.int64),
        (np.int32, np.int64),
        (np.int64, np.int64),
        # Pandas always converts int numbers to np.int64
        (np.uint, np.int64),
        (np.uint, np.int64),
        (np.uint8, np.int64),
        (np.uint16, np.int64),
        (np.uint32, np.int64),
        (np.uint64, np.int64),
        (np.bool, np.bool),
        (np.str, np.str)
        # np.object, np.void and bytes are not tested
    )

    for valid_type in valid_types:
        try:
            helper_type_validation(valid_type[0], valid_type[1])
        except:  # pylint: disable=bare-except
            # No exceptions since it should cover all exceptions for debug
            # purpose
            # Rerun test with debug inforation
            print(f"Error on types: {valid_type}")
            helper_type_validation(valid_type[0], valid_type[1], True)

    # Examples of types comparisons, which shall fail
    invalid_types = (
        (np.complex, np.int),
        (np.int, np.complex),
        (float, np.complex),
        (np.complex, float),
        (np.int, np.float),
        (np.uint8, np.float),
        (np.complex, str),
    )
    for invalid_type in invalid_types:
        with pytest.raises(SchemaError):
            helper_type_validation(invalid_type[0], invalid_type[1])

    PandasDtype.from_numpy_type(np.float)
    with pytest.raises(TypeError):
        PandasDtype.from_numpy_type(pd.DatetimeIndex)


def test_datetime():
    """Test datetime types can be validated properly by schema.validate"""
    schema = DataFrameSchema(
        columns={
            "col": Column(
                pa.DateTime,
                checks=Check(lambda s: s.min() > pd.Timestamp("2015")),
            )
        }
    )

    validated_df = schema.validate(
        pd.DataFrame(
            {"col": pd.to_datetime(["2019/01/01", "2018/05/21", "2016/03/10"])}
        )
    )

    assert isinstance(validated_df, pd.DataFrame)

    with pytest.raises(SchemaError):
        schema.validate(pd.DataFrame({"col": pd.to_datetime(["2010/01/01"])}))


@pytest.mark.skipif(
    PANDAS_VERSION.release < (1, 0, 0),  # type: ignore
    reason="pandas >= 1.0.0 required",
)
def test_pandas_extension_types():
    """Test pandas extension data type happy path."""
    # pylint: disable=no-member
    test_params = [
        (
            pd.CategoricalDtype(),
            pd.Series(["a", "a", "b", "b", "c", "c"], dtype="category"),
            None,
        ),
        (
            pd.DatetimeTZDtype(tz="UTC"),
            pd.Series(
                pd.date_range(start="20200101", end="20200301"),
                dtype="datetime64[ns, utc]",
            ),
            None,
        ),
        (pd.Int64Dtype(), pd.Series(range(10), dtype="Int64"), None),
        (
            pd.StringDtype(),
            pd.Series(["foo", "bar", "baz"], dtype="string"),
            None,
        ),
        (
            pd.PeriodDtype(freq="D"),
            pd.Series(pd.period_range("1/1/2019", "1/1/2020", freq="D")),
            None,
        ),
        (
            pd.SparseDtype("float"),
            pd.Series(range(100))
            .where(lambda s: s < 5, other=np.nan)
            .astype("Sparse[float]"),
            {"nullable": True},
        ),
        (pd.BooleanDtype(), pd.Series([1, 0, 0, 1, 1], dtype="boolean"), None),
        (
            pd.IntervalDtype(subtype="int64"),
            pd.Series(pd.IntervalIndex.from_breaks([0, 1, 2, 3, 4])),
            None,
        ),
    ]
    for dtype, data, series_kwargs in test_params:
        series_kwargs = {} if series_kwargs is None else series_kwargs
        series_schema = SeriesSchema(pandas_dtype=dtype, **series_kwargs)
        assert isinstance(series_schema.validate(data), pd.Series)


def test_python_builtin_types():
    """Test support python data types can be used for validation."""
    schema = DataFrameSchema(
        {
            "int_col": Column(int),
            "float_col": Column(float),
            "str_col": Column(str),
            "bool_col": Column(bool),
            "object_col": Column(object),
            "complex_col": Column(complex),
        }
    )
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": list("abc"),
            "bool_col": [True, False, True],
            "object_col": [[1], 1, {"foo": "bar"}],
            "complex_col": [complex(1), complex(2), complex(3)],
        }
    )
    assert isinstance(schema(df), pd.DataFrame)
    assert schema.dtype["int_col"] == PandasDtype.Int.str_alias
    assert schema.dtype["float_col"] == PandasDtype.Float.str_alias
    assert schema.dtype["str_col"] == PandasDtype.String.str_alias
    assert schema.dtype["bool_col"] == PandasDtype.Bool.str_alias
    assert schema.dtype["object_col"] == PandasDtype.Object.str_alias
    assert schema.dtype["complex_col"] == PandasDtype.Complex.str_alias


@pytest.mark.parametrize("python_type", [list, dict, set])
def test_python_builtin_types_not_supported(python_type):
    """Test unsupported python data types raise a type error."""
    with pytest.raises(TypeError):
        Column(python_type)


@pytest.mark.parametrize(
    "pandas_api_type,pandas_dtype",
    [
        ["string", PandasDtype.String],
        ["floating", PandasDtype.Float],
        ["integer", PandasDtype.Int],
        ["categorical", PandasDtype.Category],
        ["boolean", PandasDtype.Bool],
        ["datetime64", PandasDtype.DateTime],
        ["datetime", PandasDtype.DateTime],
        ["timedelta64", PandasDtype.Timedelta],
        ["timedelta", PandasDtype.Timedelta],
        ["mixed-integer", PandasDtype.Object],
    ],
)
def test_pandas_api_types(pandas_api_type, pandas_dtype):
    """Test pandas api type conversion."""
    assert PandasDtype.from_pandas_api_type(pandas_api_type) is pandas_dtype


@pytest.mark.parametrize(
    "invalid_pandas_api_type",
    [
        "foo",
        "bar",
        "baz",
        "this is not a type",
    ],
)
def test_pandas_api_type_exception(invalid_pandas_api_type):
    """Test unsupported values for pandas api type conversion."""
    with pytest.raises(TypeError):
        PandasDtype.from_pandas_api_type(invalid_pandas_api_type)


@pytest.mark.parametrize(
    "pandas_dtype", (pandas_dtype for pandas_dtype in PandasDtype)
)
def test_pandas_dtype_equality(pandas_dtype):
    """Test __eq__ implementation."""
    assert pandas_dtype is not None  # pylint:disable=singleton-comparison
    assert pandas_dtype == pandas_dtype.value


@pytest.mark.parametrize("pdtype", PandasDtype)
def test_dtype_none_comparison(pdtype):
    """Test that comparing PandasDtype to None is False."""
    assert pdtype is not None


@pytest.mark.parametrize(
    "property_fn, pdtypes",
    [
        [
            lambda x: x.is_int,
            [
                PandasDtype.Int,
                PandasDtype.Int8,
                PandasDtype.Int16,
                PandasDtype.Int32,
                PandasDtype.Int64,
                PandasDtype.INT8,
                PandasDtype.INT16,
                PandasDtype.INT32,
                PandasDtype.INT64,
            ],
        ],
        [
            lambda x: x.is_nullable_int,
            [
                PandasDtype.INT8,
                PandasDtype.INT16,
                PandasDtype.INT32,
                PandasDtype.INT64,
            ],
        ],
        [
            lambda x: x.is_nonnullable_int,
            [
                PandasDtype.Int,
                PandasDtype.Int8,
                PandasDtype.Int16,
                PandasDtype.Int32,
                PandasDtype.Int64,
            ],
        ],
        [
            lambda x: x.is_uint,
            [
                PandasDtype.UInt8,
                PandasDtype.UInt16,
                PandasDtype.UInt32,
                PandasDtype.UInt64,
                PandasDtype.UINT8,
                PandasDtype.UINT16,
                PandasDtype.UINT32,
                PandasDtype.UINT64,
            ],
        ],
        [
            lambda x: x.is_nullable_uint,
            [
                PandasDtype.UINT8,
                PandasDtype.UINT16,
                PandasDtype.UINT32,
                PandasDtype.UINT64,
            ],
        ],
        [
            lambda x: x.is_nonnullable_uint,
            [
                PandasDtype.UInt8,
                PandasDtype.UInt16,
                PandasDtype.UInt32,
                PandasDtype.UInt64,
            ],
        ],
        [
            lambda x: x.is_float,
            [
                PandasDtype.Float,
                PandasDtype.Float16,
                PandasDtype.Float32,
                PandasDtype.Float64,
            ],
        ],
        [
            lambda x: x.is_complex,
            [
                PandasDtype.Complex,
                PandasDtype.Complex64,
                PandasDtype.Complex128,
                PandasDtype.Complex256,
            ],
        ],
        [lambda x: x.is_bool, [PandasDtype.Bool]],
        [lambda x: x.is_string, [PandasDtype.String, PandasDtype.String]],
        [lambda x: x.is_category, [PandasDtype.Category]],
        [lambda x: x.is_datetime, [PandasDtype.DateTime]],
        [lambda x: x.is_timedelta, [PandasDtype.Timedelta]],
        [lambda x: x.is_object, [PandasDtype.Object]],
        [
            lambda x: x.is_continuous,
            [
                PandasDtype.Int,
                PandasDtype.Int8,
                PandasDtype.Int16,
                PandasDtype.Int32,
                PandasDtype.Int64,
                PandasDtype.INT8,
                PandasDtype.INT16,
                PandasDtype.INT32,
                PandasDtype.INT64,
                PandasDtype.UInt8,
                PandasDtype.UInt16,
                PandasDtype.UInt32,
                PandasDtype.UInt64,
                PandasDtype.UINT8,
                PandasDtype.UINT16,
                PandasDtype.UINT32,
                PandasDtype.UINT64,
                PandasDtype.Float,
                PandasDtype.Float16,
                PandasDtype.Float32,
                PandasDtype.Float64,
                PandasDtype.Complex,
                PandasDtype.Complex64,
                PandasDtype.Complex128,
                PandasDtype.Complex256,
                PandasDtype.DateTime,
                PandasDtype.Timedelta,
            ],
        ],
    ],
)
def test_dtype_is_checks(property_fn, pdtypes):
    """Test all the pandas dtype is_* properties."""
    for pdtype in pdtypes:
        assert property_fn(pdtype)


def test_category_dtype_exception():
    """Test that category dtype has no numpy dtype equivalent."""
    with pytest.raises(TypeError):
        # pylint: disable=pointless-statement
        PandasDtype.Category.numpy_dtype
