# pylint: skip-file
"""Unit tests for static type checking of dataframes.

This test module uses https://github.com/davidfritzsche/pytest-mypy-testing to
run statically check the functions marked pytest.mark.mypy_testing
"""

from typing import cast

import pandas as pd

import pandera as pa
from pandera.typing import DataFrame, Series


class Schema(pa.SchemaModel):
    id: Series[int]
    name: Series[str]


class SchemaOut(pa.SchemaModel):
    age: Series[int]


class AnotherSchema(pa.SchemaModel):
    id: Series[int]
    first_name: Series[str]


pd_df = pd.DataFrame({"id": [1], "name": ["foo"]})
valid_df = DataFrame[Schema]({"id": [1], "name": ["foo"]})
another_df = DataFrame[AnotherSchema]({"id": [1], "first_name": ["foo"]})


def fn(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[SchemaOut])


def fn_pipe_incorrect_type(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[AnotherSchema])  # mypy error


def fn_assign_copy(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30)  # mypy error


fn(valid_df)
fn(pd_df)  # mypy error
fn(another_df)  # mypy error


def fn_mutate_inplace(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    out = df.assign(age=30).pipe(DataFrame[SchemaOut])
    out.drop(["age"], axis=1, inplace=True)
    return out  # okay for mypy, pandera raises error


def fn_assign_and_get_index(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(foo=30).iloc[:3]  # okay for mypy, pandera raises error


def fn_cast_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return cast(DataFrame[SchemaOut], df)  # okay for mypy
