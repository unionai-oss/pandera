"""Utility functions for pyspark validation."""

from functools import lru_cache
from typing import List, NamedTuple, Tuple, Type, Union

import pyspark.sql.types as pst
from pyspark.sql import DataFrame

from pandera.api.checks import Check
from pandera.dtypes import DataType

CheckList = Union[Check, List[Check]]

PysparkDefaultTypes = Union[
    pst.BooleanType,
    pst.StringType,
    pst.IntegerType,
    pst.DecimalType,
    pst.FloatType,
    pst.DateType,
    pst.TimestampType,
    pst.DoubleType,
    pst.ShortType,
    pst.ByteType,
    pst.LongType,
    pst.BinaryType,
]

PySparkDtypeInputTypes = Union[
    str,
    int,
    float,
    bool,
    type,
    DataType,
    Type,
    pst.BooleanType,
    pst.StringType,
    pst.IntegerType,
    pst.DecimalType,
    pst.FloatType,
    pst.DateType,
    pst.TimestampType,
    pst.DoubleType,
    pst.ShortType,
    pst.ByteType,
    pst.LongType,
    pst.BinaryType,
]

SupportedTypes = NamedTuple(
    "SupportedTypes",
    (("table_types", Tuple[type, ...]),),
)


class PysparkDataframeColumnObject(NamedTuple):
    """Pyspark Object which holds dataframe and column value in a named tuble"""

    dataframe: DataFrame
    column_name: str


@lru_cache(maxsize=None)
def supported_types() -> SupportedTypes:
    """Get the types supported by pandera schemas."""
    # pylint: disable=import-outside-toplevel
    table_types = [DataFrame]

    try:
        table_types.append(DataFrame)

    except ImportError:  # pragma: no cover
        pass

    return SupportedTypes(
        tuple(table_types),
    )


def is_table(obj):
    """Verifies whether an object is table-like.

    Where a table is a 2-dimensional data matrix of rows and columns, which
    can be indexed in multiple different ways.
    """
    return isinstance(obj, supported_types().table_types)


def is_bool(x):
    """Verifies whether an object is a boolean type."""
    return isinstance(x, (bool, type(pst.BooleanType())))
