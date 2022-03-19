"""Utility functions for validation."""

from functools import lru_cache
from typing import NamedTuple, Optional, Tuple, Union

import pandas as pd

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
def _supported_types():
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
    return isinstance(obj, _supported_types().table_types)


def is_field(obj):
    """Verifies whether an object is field-like.

    Where a field is a columnar representation of data in a table-like
    data structure.
    """
    return isinstance(obj, _supported_types().field_types)


def is_index(obj):
    """Verifies whether an object is a table index."""
    return isinstance(obj, _supported_types().index_types)


def is_multiindex(obj):
    """Verifies whether an object is a multi-level table index."""
    return isinstance(obj, _supported_types().multiindex_types)


def is_supported_check_obj(obj):
    """Verifies whether an object is table- or field-like."""
    return is_table(obj) or is_field(obj)


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
        # NOTE: this is a hack to support pyspark.pandas and modin, since you
        # can't use groupby on a dataframe with another dataframe
        if type(failure_cases).__module__.startswith("pyspark.pandas") or type(
            failure_cases
        ).__module__.startswith("modin.pandas"):
            failure_cases = (
                failure_cases.rename("failure_cases")
                .to_frame()
                .assign(check_output=check_output)
                .groupby("check_output")
                .head(n_failure_cases)["failure_cases"]
            )
        else:
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
