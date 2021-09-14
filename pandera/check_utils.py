"""Utility functions for validation."""

from typing import Optional, Tuple, Union

import pandas as pd

TABLE_TYPES = [pd.DataFrame]
FIELD_TYPES = [pd.Series]
INDEX_TYPES = [pd.Index]
MULTIINDEX_TYPES = [pd.MultiIndex]

try:
    import databricks.koalas as ks

    TABLE_TYPES.append(ks.DataFrame)
    FIELD_TYPES.append(ks.Series)
    INDEX_TYPES.append(ks.Index)
    MULTIINDEX_TYPES.append(ks.MultiIndex)
except ImportError:
    pass
try:
    import modin.pandas as mpd

    TABLE_TYPES.append(mpd.DataFrame)
    FIELD_TYPES.append(mpd.Series)
    INDEX_TYPES.append(mpd.Index)
    MULTIINDEX_TYPES.append(mpd.MultiIndex)
except ImportError:
    pass


TABLE_TYPES = tuple(TABLE_TYPES)
FIELD_TYPES = tuple(FIELD_TYPES)
INDEX_TYPES = tuple(INDEX_TYPES)
MULTIINDEX_TYPES = tuple(MULTIINDEX_TYPES)


def is_table(obj):
    """Verifies whether an object is table-like.

    Where a table is a 2-dimensional data matrix of rows and columns, which
    can be indexed in multiple different ways.
    """
    return isinstance(obj, TABLE_TYPES)


def is_field(obj):
    """Verifies whether an object is field-like.

    Where a field is a columnar representation of data in a table-like
    data structure.
    """
    return isinstance(obj, FIELD_TYPES)


def is_index(obj):
    """Verifies whether an object is a table index."""
    return isinstance(obj, INDEX_TYPES)


def is_multiindex(obj):
    """Verifies whether an object is a multi-level table index."""
    return isinstance(obj, MULTIINDEX_TYPES)


def is_supported_check_obj(obj):
    """Verifies whether an object is table- or field-like."""
    return (
        is_table(obj) or is_field(obj) or is_index(obj) or is_multiindex(obj)
    )


def prepare_series_check_output(
    check_obj: Union[pd.Series, pd.DataFrame],
    check_output: pd.Series,
    ignore_na: bool = True,
    n_failure_cases: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Prepare the check output and failure cases for a Series check output.

    check_obj can be a dataframe, since a check function can potentially return
    a Series resulting from applying some check function that outputs a Series.
    """
    if ignore_na:
        isna = (
            check_obj.isna().any(axis="columns")
            if isinstance(check_obj, pd.DataFrame)
            else check_obj.isna()
        )
        check_output = check_output | isna
    failure_cases = check_obj[~check_output]
    if not failure_cases.empty and n_failure_cases is not None:
        failure_cases = failure_cases.groupby(check_output).head(
            n_failure_cases
        )
    return check_output, failure_cases


def prepare_dataframe_check_output(
    check_obj: pd.DataFrame,
    check_output: pd.DataFrame,
    df_orig: Optional[pd.DataFrame] = None,
    ignore_na: bool = True,
    n_failure_cases: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Unstack a dataframe of boolean values.

    Check results consisting of a boolean dataframe should be reported at the
    most granular level.
    """
    if df_orig is not None:
        assert df_orig.shape == check_output.shape

    if df_orig is None:
        df_orig = check_obj
    check_output = check_output.unstack()
    if ignore_na:
        check_output = check_output | df_orig.unstack().isna()
    failure_cases = (
        check_obj.unstack()[~check_output]
        .rename("failure_case")
        .rename_axis(["column", "index"])
        .reset_index()
    )
    if not failure_cases.empty and n_failure_cases is not None:
        failure_cases = failure_cases.drop_duplicates().head(n_failure_cases)
    return check_output, failure_cases
