"""Utility functions for pyspark validation."""

from functools import lru_cache
from typing import List, NamedTuple, Tuple, Type, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore [misc]

from pandera.api.checks import Check
from pandera.dtypes import DataType
from pyspark.sql import DataFrame
import pyspark.sql.types as pst

CheckList = Union[Check, List[Check]]

PySparkDtypeInputTypes = Union[
    str,
    int,
    float,
    bool,
    type,
    DataType,
    Type,
]

StrictType = Union[bool, Literal["filter"]]

SupportedTypes = NamedTuple(
    "SupportedTypes",
    (("table_types", Tuple[type, ...]),),
)


@lru_cache(maxsize=None)
def supported_types() -> SupportedTypes:
    """Get the types supported by pandera schemas."""
    # pylint: disable=import-outside-toplevel
    table_types = [DataFrame]

    try:

        table_types.append(DataFrame)

    except ImportError:
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
    return isinstance(x, (bool, pst.BooleanType()))
