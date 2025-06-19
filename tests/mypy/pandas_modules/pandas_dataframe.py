# pylint: skip-file
"""Unit tests for static type checking of dataframes.

This test module uses https://github.com/davidfritzsche/pytest-mypy-testing to
run statically check the functions marked pytest.mark.mypy_testing
"""

from typing import Optional, cast

import pandas as pd

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Schema(pa.DataFrameModel):
    id: Series[int]
    name: Series[str]


class SchemaOut(pa.DataFrameModel):
    age: Series[int]


class AnotherSchema(pa.DataFrameModel):
    id: Series[int]
    first_name: Optional[Series[str]]


def fn(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[SchemaOut])  # mypy okay


def fn_pipe_incorrect_type(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[AnotherSchema])  # mypy error
    # error: Argument 1 to "pipe" of "NDFrame" has incompatible type "Type[DataFrame[Any]]";  # noqa
    # expected "Union[Callable[..., DataFrame[SchemaOut]], Tuple[Callable[..., DataFrame[SchemaOut]], str]]"  [arg-type]  # noqa


def fn_assign_copy(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30)  # mypy error
    # error: Incompatible return value type (got "pandas.core.frame.DataFrame",
    # expected "pandera.typing.pandas.DataFrame[SchemaOut]")  [return-value]


# Define a few dataframe objects
schema_df = DataFrame[Schema]({"id": [1], "name": ["foo"]})
pandas_df = pd.DataFrame({"id": [1], "name": ["foo"]})
another_df = DataFrame[AnotherSchema]({"id": [1], "first_name": ["foo"]})


fn(schema_df)  # mypy okay

fn(pandas_df)  # mypy error
# error: Argument 1 to "fn" has incompatible type "pandas.core.frame.DataFrame";  # noqa
# expected "pandera.typing.pandas.DataFrame[Schema]"  [arg-type]

fn(another_df)  # mypy error
# error: Argument 1 to "fn" has incompatible type "DataFrame[AnotherSchema]";
# expected "DataFrame[Schema]"  [arg-type]


def fn_pipe_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(age=30).pipe(DataFrame[SchemaOut])  # mypy okay


def fn_cast_dataframe(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return cast(DataFrame[SchemaOut], df.assign(age=30))  # mypy okay


@pa.check_types
def fn_mutate_inplace(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    out = df.assign(age=30).pipe(DataFrame[SchemaOut])
    out.drop(columns="age", inplace=True)
    return out  # okay for mypy, pandera raises error


@pa.check_types
def fn_assign_and_get_index(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return df.assign(foo=30).iloc[:3]  # mypy error
    # error: Incompatible return value type (got "pandas.core.frame.DataFrame",
    # expected "pandera.typing.pandas.DataFrame[SchemaOut]")  [return-value]


@pa.check_types
def fn_cast_dataframe_invalid(df: DataFrame[Schema]) -> DataFrame[SchemaOut]:
    return cast(
        DataFrame[SchemaOut], df
    )  # okay for mypy, pandera raises error
