import os
from unittest.mock import MagicMock

import databricks.koalas as ks
import pytest

import pandera as pa

try:
    import hypothesis
    import hypothesis.extra.numpy as npst
    import hypothesis.strategies as st
except ImportError:
    HAS_HYPOTHESIS = False
    hypothesis = MagicMock()
    st = MagicMock()
else:
    HAS_HYPOTHESIS = True


os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


@pytest.mark.parametrize("coerce", [True, False])
@hypothesis.given(st.data())
def _test_dataframe_schema(data):
    # TODO:
    # tests should be parameterized on:
    # - all data types
    pa.engines.pandas_engine.Engine.get_registered_dtypes()
    # - all checks
    # get all classmethods of subclass pa.Check to iterate through all checks.
    pass


@pytest.mark.parametrize("coerce", [True, False])
def test_dataframe_schema(coerce):
    schema = pa.DataFrameSchema(
        {
            "int_column": pa.Column(int, pa.Check.ge(0)),
            "float_column": pa.Column(float, pa.Check.le(0)),
            "str_column": pa.Column(str, pa.Check.isin(list("abcde"))),
        },
        coerce=coerce,
    )
    kdf = ks.DataFrame(
        {
            "int_column": range(10),
            "float_column": [float(-x) for x in range(10)],
            "str_column": list("aabbcceedd"),
        }
    )

    # TODO: look at type support in koalas
    # https://koalas.readthedocs.io/en/latest/user_guide/types.html#type-casting-between-pandas-and-koalas

    assert isinstance(schema.validate(kdf), ks.DataFrame)
