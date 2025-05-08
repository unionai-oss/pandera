"""Test for forward references when __future__.annotations is enabled."""

from __future__ import annotations

import typing

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

import pandera.pandas as pa


# define check_types-decorated function that uses forward reference to Schema
@pa.check_types
def func(data: pa.typing.DataFrame[Model]) -> pa.typing.DataFrame[Model]:
    return data


class Model(pa.DataFrameModel):
    a: int


def test_simple_forwardref():

    func(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(pa.errors.SchemaError):
        func(pd.DataFrame({"a": [*"abc"]}))


def test_forwardref_with_local_model():
    """
    Test that forward references with a local model fails because the underlying
    model is not available at the time of run-time type evaluation.
    """

    @pa.check_types
    def local_func(
        data: pa.typing.DataFrame[LocalModel],
    ) -> pa.typing.DataFrame[LocalModel]:
        return data

    class LocalModel(pa.DataFrameModel):
        a: int

    with pytest.raises(NameError):
        local_func(pd.DataFrame({"a": [1, 2, 3]}))


# define a model that uses a custom type
class ModelWithCustomType(pa.DataFrameModel):
    a: int
    b: float
    c: typing.Annotated[str, CustomType]


class CustomType: ...


def test_forwardref_in_df_model_with_custom_type():
    schema = ModelWithCustomType.to_schema()
    assert schema == pa.DataFrameSchema(
        name="ModelWithCustomType",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(float),
            "c": pa.Column(str),
        },
    )


def test_forwardref_in_df_model_with_annotated_types():

    class ModelWithAnnotatedType(pa.DataFrameModel):
        a: int
        b: typing.Annotated[int, "foo"]

    schema = ModelWithAnnotatedType.to_schema()
    assert schema == pa.DataFrameSchema(
        name="ModelWithAnnotatedType",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(int),
        },
    )


def test_forwardref_with_local_custom_types():
    class LocalModelWithCustomType(pa.DataFrameModel):
        a: int
        b: float
        c: typing.Annotated[str, CustomType]
        d: typing.Annotated[pd.DatetimeTZDtype, "ns", "utc"]

    class CustomType: ...

    schema = LocalModelWithCustomType.to_schema()
    assert schema == pa.DataFrameSchema(
        name="LocalModelWithCustomType",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(float),
            "c": pa.Column(str),
            "d": pa.Column(pd.DatetimeTZDtype("ns", "utc")),
        },
    )


def test_forwardref_with_pandera_dataframe_generic_initialization():

    def func(data) -> pa.typing.DataFrame[LocalModel]:
        return pa.typing.DataFrame[LocalModel](data)

    class LocalModel(pa.DataFrameModel):
        a: int

    expected = pa.typing.DataFrame[LocalModel]({"a": [1, 2, 3]})
    assert_frame_equal(func({"a": [1, 2, 3]}), expected)

    with pytest.raises(pa.errors.SchemaError):
        func({"a": [*"abc"]})
