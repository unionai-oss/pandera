"""Test typing annotations for the model api."""
# pylint:disable=missing-class-docstring,too-few-public-methods
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
import pytest

import pandera as pa
from pandera.dtypes import LEGACY_PANDAS, PandasDtype
from pandera.model import SchemaModel
from pandera.typing import Series


class SchemaBool(pa.SchemaModel):
    col: Series[pa.typing.Bool]


class SchemaDateTime(pa.SchemaModel):
    col: Series[pa.typing.DateTime]


class SchemaCategory(pa.SchemaModel):
    col: Series[pa.typing.Category]


class SchemaFloat(pa.SchemaModel):
    col: Series[pa.typing.Float]


class SchemaFloat16(pa.SchemaModel):
    col: Series[pa.typing.Float16]


class SchemaFloat32(pa.SchemaModel):
    col: Series[pa.typing.Float32]


class SchemaFloat64(pa.SchemaModel):
    col: Series[pa.typing.Float64]


class SchemaInt(pa.SchemaModel):
    col: Series[pa.typing.Int]


class SchemaInt8(pa.SchemaModel):
    col: Series[pa.typing.Int8]


class SchemaUInt8(pa.SchemaModel):
    col: Series[pa.typing.UInt8]


class SchemaInt16(pa.SchemaModel):
    col: Series[pa.typing.Int16]


class SchemaUInt16(pa.SchemaModel):
    col: Series[pa.typing.UInt16]


class SchemaInt32(pa.SchemaModel):
    col: Series[pa.typing.Int32]


class SchemaUInt32(pa.SchemaModel):
    col: Series[pa.typing.UInt32]


class SchemaInt64(pa.SchemaModel):
    col: Series[pa.typing.Int64]


class SchemaUInt64(pa.SchemaModel):
    col: Series[pa.typing.UInt64]


class SchemaObject(pa.SchemaModel):
    col: Series[pa.typing.Object]


class SchemaString(pa.SchemaModel):
    col: Series[pa.typing.String]


class SchemaTimedelta(pa.SchemaModel):
    col: Series[pa.typing.Timedelta]


class SchemaINT8(pa.SchemaModel):
    col: Series[pa.typing.INT8]


class SchemaUINT8(pa.SchemaModel):
    col: Series[pa.typing.UINT8]


class SchemaINT16(pa.SchemaModel):
    col: Series[pa.typing.INT16]


class SchemaUINT16(pa.SchemaModel):
    col: Series[pa.typing.UINT16]


class SchemaINT32(pa.SchemaModel):
    col: Series[pa.typing.INT32]


class SchemaUINT32(pa.SchemaModel):
    col: Series[pa.typing.UINT32]


class SchemaINT64(pa.SchemaModel):
    col: Series[pa.typing.INT64]


class SchemaUINT64(pa.SchemaModel):
    col: Series[pa.typing.UINT64]


def _test_literal_pandas_dtype(
    model: Type[SchemaModel], pandas_dtype: PandasDtype
):
    schema = model.to_schema()
    assert (
        schema.columns["col"].dtype
        == pa.Column(pandas_dtype, name="col").dtype
    )


@pytest.mark.parametrize(
    "model, pandas_dtype",
    [
        (SchemaBool, pa.Bool),
        (SchemaDateTime, pa.DateTime),
        (SchemaCategory, pa.Category),
        (SchemaFloat, pa.Float),
        (SchemaFloat16, pa.Float16),
        (SchemaFloat32, pa.Float32),
        (SchemaFloat64, pa.Float64),
        (SchemaInt, pa.Int),
        (SchemaInt8, pa.Int8),
        (SchemaInt16, pa.Int16),
        (SchemaInt32, pa.Int32),
        (SchemaInt64, pa.Int64),
        (SchemaUInt8, pa.UInt8),
        (SchemaUInt16, pa.UInt16),
        (SchemaUInt32, pa.UInt32),
        (SchemaUInt64, pa.UInt64),
        (SchemaObject, pa.Object),
        (SchemaString, pa.String),
        (SchemaTimedelta, pa.Timedelta),
    ],
)
def test_literal_legacy_pandas_dtype(
    model: Type[SchemaModel], pandas_dtype: PandasDtype
):
    """Test literal annotations with the legacy pandas dtypes."""
    _test_literal_pandas_dtype(model, pandas_dtype)


@pytest.mark.skipif(LEGACY_PANDAS, reason="pandas >= 1.0.0 required")
@pytest.mark.parametrize(
    "model, pandas_dtype",
    [
        (SchemaUINT8, pa.UINT8),
        (SchemaUINT16, pa.UINT16),
        (SchemaUINT32, pa.UINT32),
        (SchemaUINT64, pa.UINT64),
        (SchemaUINT8, pa.UINT8),
        (SchemaUINT16, pa.UINT16),
        (SchemaUINT32, pa.UINT32),
        (SchemaUINT64, pa.UINT64),
    ],
)
def test_literal_new_pandas_dtype(
    model: Type[SchemaModel], pandas_dtype: PandasDtype
):
    """Test literal annotations with the new nullable pandas dtypes."""
    _test_literal_pandas_dtype(model, pandas_dtype)


class SchemaCategoricalDtypeClass(pa.SchemaModel):
    col: Series[pd.CategoricalDtype]


class SchemaDatetimeTZDtypeClass(pa.SchemaModel):
    col: Series[pd.DatetimeTZDtype]


class SchemaIntervalDtypeClass(pa.SchemaModel):
    col: Series[pd.IntervalDtype]


class SchemaPeriodDtypeClass(pa.SchemaModel):
    col: Series[pd.PeriodDtype]


class SchemaSparseDtypeClass(pa.SchemaModel):
    col: Series[pd.SparseDtype]


def _test_pandas_extension_dtype(
    model: Type[SchemaModel], dtype: Type, dtype_kwargs: Dict[str, Any] = None
):
    dtype_kwargs = dtype_kwargs or {}
    schema = model.to_schema()

    actual = schema.columns["col"].dtype
    expected = pa.Column(dtype(**dtype_kwargs), name="col").dtype
    assert actual == expected


def _test_pandas_extension_dtype_class(
    model: Type[SchemaModel], dtype: Type, has_mandatory_args: bool
):
    if has_mandatory_args:
        err_msg = "cannot be instantiated"
        with pytest.raises(TypeError, match=err_msg):
            model.to_schema()
    else:
        _test_pandas_extension_dtype(model, dtype)


@pytest.mark.parametrize(
    "model, dtype, has_mandatory_args",
    [
        (SchemaCategoricalDtypeClass, pd.CategoricalDtype, False),
        # DatetimeTZDtype: tz is implictly required
        (SchemaDatetimeTZDtypeClass, pd.DatetimeTZDtype, True),
        (SchemaIntervalDtypeClass, pd.IntervalDtype, False),
        # PeriodDtype: freq is implicitely required -> str(pd.PeriodDtype()) raises AttributeError
        (SchemaPeriodDtypeClass, pd.PeriodDtype, True),
        (SchemaSparseDtypeClass, pd.SparseDtype, False),
    ],
)
def test_legacy_pandas_extension_dtype_class(
    model, dtype: pd.core.dtypes.base.ExtensionDtype, has_mandatory_args: bool
):
    """Test type annotations for legacy pandas extension dtypes."""
    _test_pandas_extension_dtype_class(model, dtype, has_mandatory_args)


class SchemaCategoricalDtypeDefault(pa.SchemaModel):
    col: Series[pa.typing.CategoricalDtype]


class SchemaDatetimeTZDtypeDefault(pa.SchemaModel):
    col: Series[pa.typing.DatetimeTZDtype]


class SchemaIntervalDtypeDefault(pa.SchemaModel):
    col: Series[pa.typing.IntervalDtype]


class SchemaPeriodDtypeDefault(pa.SchemaModel):
    col: Series[pa.typing.PeriodDtype]


class SchemaSparseDtypeDefault(pa.SchemaModel):
    col: Series[pa.typing.SparseDtype]


class SchemaCategoricalDtypeParams(pa.SchemaModel):
    col: Series[pa.typing.CategoricalDtype[("b", "a"), True]]


class SchemaDatetimeTZDtypeParams(pa.SchemaModel):
    col: Series[pa.typing.DatetimeTZDtype["ns", "est"]]


class SchemaIntervalDtypeParams(pa.SchemaModel):
    col: Series[pa.typing.IntervalDtype["int32"]]


class SchemaPeriodDtypeParams(pa.SchemaModel):
    col: Series[pa.typing.PeriodDtype["D"]]


class SchemaSparseDtypeParams(pa.SchemaModel):
    col: Series[pa.typing.SparseDtype[np.int32, 0]]


@pytest.mark.parametrize(
    "model, dtype, dtype_kwargs",
    [
        (
            SchemaCategoricalDtypeDefault,
            pd.CategoricalDtype,
            None,
        ),
        (
            SchemaDatetimeTZDtypeDefault,
            pd.DatetimeTZDtype,
            {"unit": "ns", "tz": "UTC"},
        ),
        (SchemaIntervalDtypeDefault, pd.IntervalDtype, None),
        # PeriodDtype: freq is implicitely required -> str(pd.PeriodDtype()) raises AttributeError
        # (SchemaPeriodDtypeDefault, pd.PeriodDtype, None),
        (SchemaSparseDtypeDefault, pd.SparseDtype, None),
        (
            SchemaCategoricalDtypeParams,
            pd.CategoricalDtype,
            {"categories": ["b", "a"], "ordered": True},
        ),
        (
            SchemaDatetimeTZDtypeParams,
            pd.DatetimeTZDtype,
            {"unit": "ns", "tz": "EST"},
        ),
        (SchemaIntervalDtypeParams, pd.IntervalDtype, {"subtype": "int32"}),
        (SchemaPeriodDtypeParams, pd.PeriodDtype, {"freq": "D"}),
        (
            SchemaSparseDtypeParams,
            pd.SparseDtype,
            {"dtype": np.int32, "fill_value": 0},
        ),
    ],
)
def test_pandas_extension_dtype(
    model: Type[SchemaModel], dtype: Type, dtype_kwargs: Dict[str, Any]
):
    """Test type annotations for pandas extension dtypes defined with pandera's internal
    typing module."""
    _test_pandas_extension_dtype(model, dtype, dtype_kwargs)


if not LEGACY_PANDAS:

    class SchemaInt8Dtype(pa.SchemaModel):
        col: Series[pd.Int8Dtype]

    class SchemaInt16Dtype(pa.SchemaModel):
        col: Series[pd.Int16Dtype]

    class SchemaInt32Dtype(pa.SchemaModel):
        col: Series[pd.Int32Dtype]

    class SchemaInt64Dtype(pa.SchemaModel):
        col: Series[pd.Int64Dtype]

    class SchemaUInt8Dtype(pa.SchemaModel):
        col: Series[pd.UInt8Dtype]

    class SchemaUInt16Dtype(pa.SchemaModel):
        col: Series[pd.UInt16Dtype]

    class SchemaUInt32Dtype(pa.SchemaModel):
        col: Series[pd.UInt32Dtype]

    class SchemaUInt64Dtype(pa.SchemaModel):
        col: Series[pd.UInt64Dtype]

    class SchemaStringDtype(pa.SchemaModel):
        col: Series[pd.StringDtype]

    class SchemaBooleanDtype(pa.SchemaModel):
        col: Series[pd.BooleanDtype]

    @pytest.mark.skipif(LEGACY_PANDAS, reason="pandas >= 1.0.0 required")
    @pytest.mark.parametrize(
        "model, dtype, has_mandatory_args",
        [
            (SchemaInt8Dtype, pd.Int8Dtype, False),
            (SchemaInt16Dtype, pd.Int16Dtype, False),
            (SchemaInt32Dtype, pd.Int32Dtype, False),
            (SchemaInt64Dtype, pd.Int64Dtype, False),
            (SchemaUInt8Dtype, pd.UInt8Dtype, False),
            (SchemaUInt16Dtype, pd.UInt16Dtype, False),
            (SchemaUInt32Dtype, pd.UInt32Dtype, False),
            (SchemaUInt64Dtype, pd.UInt64Dtype, False),
            (SchemaStringDtype, pd.StringDtype, False),
            (SchemaBooleanDtype, pd.BooleanDtype, False),
        ],
    )
    def test_new_pandas_extension_dtype_class(
        model,
        dtype: pd.core.dtypes.base.ExtensionDtype,
        has_mandatory_args: bool,
    ):
        """Test type annotations with the new nullable pandas dtypes."""
        _test_pandas_extension_dtype_class(model, dtype, has_mandatory_args)
