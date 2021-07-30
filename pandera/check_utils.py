"""Utility functions for validation."""

from typing import Optional, Tuple, Union

import pandas as pd


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
