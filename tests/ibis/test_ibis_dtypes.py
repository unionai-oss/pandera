"""Ibis dtype tests."""

import ibis
import ibis.expr.datatypes as dt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ibis import _
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

import pandera.ibis as pa
from pandera.engines import ibis_engine as ie

NUMERIC_TYPES = [
    ie.Int32,
    ie.Int64,
    ie.Float64,
]

TEMPORAL_TYPES = []  # type: ignore # TODO(deepyaman): Delete annotation once populated.

OTHER_TYPES = [
    ie.String,
]

SPECIAL_TYPES = [
    ie.Map,
]

ALL_TYPES = NUMERIC_TYPES + TEMPORAL_TYPES + OTHER_TYPES


@st.composite
def memtables(draw, **kwargs):
    return ibis.memtable(draw(dataframes(**kwargs)))


def get_table_strategy(dtype: dt.DataType) -> st.SearchStrategy:
    """Get a strategy for an Ibis table of a given dtype."""
    return memtables(
        cols=2,
        allowed_dtypes=dtype.to_polars(),
        allow_null=False,
        min_size=10,
        max_size=10,
    )


@pytest.mark.parametrize("dtype", ALL_TYPES)
@given(st.data())
@settings(max_examples=10)
def test_coerce_no_cast(dtype, data):
    """Test that dtypes can be coerced without casting."""
    pandera_dtype = dtype()
    t = data.draw(get_table_strategy(dtype=pandera_dtype.type))
    coerced = pandera_dtype.coerce(t)
    assert_frame_equal(t.to_polars(), coerced.to_polars())


@pytest.mark.parametrize(
    "from_dtype, to_dtype, strategy",
    [
        (ie.UInt32(), ie.UInt64(), get_table_strategy),
        (ie.Float32(), ie.Float64(), get_table_strategy),
        (ie.Int16(), ie.String(), get_table_strategy),
    ],
)
@given(st.data())
@settings(max_examples=5)
def test_coerce_cast(from_dtype, to_dtype, strategy, data):
    """Test that dtypes can be coerced with casting."""
    s = data.draw(strategy(from_dtype.type))

    coerced = to_dtype.coerce(data_container=s)
    for dtype in coerced.schema().values():
        assert dtype == to_dtype.type


@pytest.mark.parametrize("dtype", ALL_TYPES + SPECIAL_TYPES)
def test_check_not_equivalent(dtype):
    """Test that check() rejects non-equivalent dtypes."""
    if str(ie.Engine.dtype(dtype)) == "string":
        actual_dtype = ie.Engine.dtype(int)
    else:
        actual_dtype = ie.Engine.dtype(str)
    expected_dtype = ie.Engine.dtype(dtype)
    assert not actual_dtype.check(expected_dtype)


@pytest.mark.parametrize("dtype", ALL_TYPES + SPECIAL_TYPES)
def test_check_equivalent(dtype):
    """Test that check() accepts equivalent dtypes."""
    actual_dtype = ie.Engine.dtype(dtype)
    expected_dtype = ie.Engine.dtype(dtype)
    assert actual_dtype.check(expected_dtype)


@pytest.mark.parametrize(
    "first_dtype, second_dtype, equivalent",
    [
        (ie.Int8, ie.Int16, False),
        (ie.DateTime(), ie.Date, False),
        (
            ie.DateTime(timezone=None, scale=1),
            ie.DateTime(timezone=None, scale=2),
            False,
        ),
        (
            ie.DateTime(timezone=None, scale=1),
            ie.DateTime(timezone=None, scale=1),
            True,
        ),
        (ie.Timedelta(unit="us"), ie.Timedelta(unit="ns"), False),
        (ie.Timedelta(unit="us"), ie.Timedelta(unit="us"), True),
    ],
)
def test_check_equivalent_custom(first_dtype, second_dtype, equivalent):
    """Test that check() rejects non-equivalent dtypes."""
    first_engine_dtype = ie.Engine.dtype(first_dtype)
    second_engine_dtype = ie.Engine.dtype(second_dtype)
    assert first_engine_dtype.check(second_engine_dtype) is equivalent


@pytest.mark.parametrize(
    "ibis_dtype, expected_dtype",
    [
        (dt.Decimal(5, 2), ie.Decimal(5, 2)),
        (dt.Decimal(None, 2), ie.Decimal(38, 2)),
        (dt.Decimal(5, None), ie.Decimal(5, 0)),
        (dt.Decimal(None, None), ie.Decimal(38, 0)),
    ],
)
def test_ibis_decimal_from_parametrized_dtype(ibis_dtype, expected_dtype):
    pandera_dtype = ie.Engine.dtype(ibis_dtype)

    assert pandera_dtype.precision == expected_dtype.precision
    assert pandera_dtype.scale == expected_dtype.scale


@pytest.mark.parametrize("key_dtype", ALL_TYPES)
@pytest.mark.parametrize("value_dtype", ALL_TYPES)
def test_ibis_map_nested_type(key_dtype, value_dtype):
    ibis_dtype = dt.Map(key_dtype.type(), value_dtype.type())
    pandera_dtype = ie.Engine.dtype(ibis_dtype)

    assert pandera_dtype.check(ibis_dtype)
    assert pandera_dtype.check(pandera_dtype)


def test_ibis_map_type():
    # https://github.com/unionai-oss/pandera/issues/2201
    data = {
        "id": ["01", "02", "03"],
        "key_col": [
            [1, 2, 3],
            [
                1,
                2,
            ],
            None,
        ],
        "value_col": [
            ["value_1A", "value_1B", "value_1C"],
            ["value_2A", "value_2B"],
            None,
        ],
    }
    df = (
        ibis.memtable(data)
        .mutate(dict_col=ibis.map(_.key_col, _.value_col))
        .drop("key_col", "value_col")
    )

    class ValidateSchema(pa.DataFrameModel):
        id: str = pa.Field(nullable=False)
        dict_col: ibis.dtype("map<int64, string>") = pa.Field(nullable=True)

    ValidateSchema.validate(df)

    class ValidateSchema(pa.DataFrameModel):
        id: str = pa.Field(nullable=False)
        dict_col: dt.Map = pa.Field(nullable=True)

    ValidateSchema.validate(df)

    class ValidateSchema(pa.DataFrameModel):
        id: str = pa.Field(nullable=False)
        dict_col: ibis.dtype("map<string, string>") = pa.Field(nullable=True)

    with pytest.raises(pa.errors.SchemaError):
        ValidateSchema.validate(df)
