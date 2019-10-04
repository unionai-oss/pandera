import pandas as pd

from pandera import Column, DataFrameSchema
from pandera import dtypes


def test_dtypes():
    for dtype in [
            dtypes.Float,
            dtypes.Float16,
            dtypes.Float32,
            dtypes.Float64]:
        schema = DataFrameSchema({"col": Column(dtype, nullable=False)})
        validated_df = schema.validate(
            pd.DataFrame(
                {"col": [-123.1, -7654.321, 1.0, 1.1, 1199.51, 5.1, 4.6]},
                dtype=dtype.value))
        assert isinstance(validated_df, pd.DataFrame)

    for dtype in [
            dtypes.Int,
            dtypes.Int8,
            dtypes.Int16,
            dtypes.Int32,
            dtypes.Int64]:
        schema = DataFrameSchema({"col": Column(dtype, nullable=False)})
        validated_df = schema.validate(
            pd.DataFrame(
                {"col": [-712, -4, -321, 0, 1, 777, 5, 123, 9000]},
                dtype=dtype.value))
        assert isinstance(validated_df, pd.DataFrame)

    for dtype in [
            dtypes.UInt8,
            dtypes.UInt16,
            dtypes.UInt32,
            dtypes.UInt64]:
        schema = DataFrameSchema({"col": Column(dtype, nullable=False)})
        validated_df = schema.validate(
            pd.DataFrame(
                {"col": [1, 777, 5, 123, 9000]}, dtype=dtype.value))
        assert isinstance(validated_df, pd.DataFrame)
