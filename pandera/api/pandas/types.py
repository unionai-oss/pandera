"""Utility functions for pandas validation."""

from functools import lru_cache
from typing import NamedTuple, Tuple, Type, Union

import numpy as np
import pandas as pd

from pandera.dtypes import DataType

PandasDtypeInputTypes = Union[
    str,
    type,
    DataType,
    Type,
    pd.core.dtypes.base.ExtensionDtype,
    np.dtype,
]

SupportedTypes = NamedTuple(
    "SupportedTypes",
    (
        ("table_types", Tuple[type, ...]),
        ("field_types", Tuple[type, ...]),
        ("index_types", Tuple[type, ...]),
        ("multiindex_types", Tuple[type, ...]),
    ),
)


@lru_cache(maxsize=None)
def supported_types() -> SupportedTypes:
    """Get the types supported by pandera schemas."""
    # pylint: disable=import-outside-toplevel
    table_types = [pd.DataFrame]
    field_types = [pd.Series]
    index_types = [pd.Index]
    multiindex_types = [pd.MultiIndex]

    try:
        import pyspark.pandas as ps

        table_types.append(ps.DataFrame)
        field_types.append(ps.Series)
        index_types.append(ps.Index)
        multiindex_types.append(ps.MultiIndex)
    except ImportError:
        pass
    try:  # pragma: no cover
        import modin.pandas as mpd

        table_types.append(mpd.DataFrame)
        field_types.append(mpd.Series)
        index_types.append(mpd.Index)
        multiindex_types.append(mpd.MultiIndex)
    except ImportError:
        pass
    try:
        import dask.dataframe as dd

        table_types.append(dd.DataFrame)
        field_types.append(dd.Series)
        index_types.append(dd.Index)
    except ImportError:
        pass

    return SupportedTypes(
        tuple(table_types),
        tuple(field_types),
        tuple(index_types),
        tuple(multiindex_types),
    )


def is_table(obj):
    """Verifies whether an object is table-like.

    Where a table is a 2-dimensional data matrix of rows and columns, which
    can be indexed in multiple different ways.
    """
    return isinstance(obj, supported_types().table_types)


def is_field(obj):
    """Verifies whether an object is field-like.

    Where a field is a columnar representation of data in a table-like
    data structure.
    """
    return isinstance(obj, supported_types().field_types)


def is_index(obj):
    """Verifies whether an object is a table index."""
    return isinstance(obj, supported_types().index_types)


def is_multiindex(obj):
    """Verifies whether an object is a multi-level table index."""
    return isinstance(obj, supported_types().multiindex_types)


def is_table_or_field(obj):
    """Verifies whether an object is table- or field-like."""
    return is_table(obj) or is_field(obj)


is_supported_check_obj = is_table_or_field


def is_bool(x):
    """Verifies whether an object is a boolean type."""
    return isinstance(x, (bool, np.bool_))
