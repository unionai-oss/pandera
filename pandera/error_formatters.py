"""Make schema error messages human-friendly."""

from typing import Union

import pandas as pd

from .checks import Check


def format_generic_error_message(
        parent_schema,
        check: Check,
        check_index: int,
) -> str:
    """Construct an error message when a check validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    """
    return "%s failed series validator %d: %s" % \
        (parent_schema, check_index, check)


def format_vectorized_error_message(
        parent_schema,
        check: Check,
        check_index: int,
        failure_cases: pd.Series) -> str:
    """Construct an error message when a validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    :param failure_cases: The failure cases encountered by the element-wise
        or vectorized validator.

    """
    return (
        "%s failed element-wise validator %d:\n"
        "%s\nfailure cases:\n%s" % (
            parent_schema,
            check_index,
            check,
            format_failure_cases(failure_cases, check.n_failure_cases)
        )
    )


def format_failure_cases(
        failure_cases: Union[pd.DataFrame, pd.Series],
        n_cases: int) -> pd.DataFrame:
    """Construct readable error messages for vectorized_error_message.

    :param failure_cases: The failure cases encountered by the element-wise
        or vectorized validator.
    :returns: DataFrame where index contains failure cases, the "index"
        column contains a list of integer indexes in the validation
        DataFrame that caused the failure, and a "count" column
        representing how many failures of that case occurred.

    """
    if hasattr(failure_cases, "index") and \
            isinstance(failure_cases.index, pd.MultiIndex):
        index_name = failure_cases.index.name
        failure_cases = (
            failure_cases
            .rename("failure_case")
            .reset_index()
            .assign(
                index=lambda df: (
                    df.apply(tuple, axis=1).astype(str)
                )
            )
        )
    elif isinstance(failure_cases, pd.DataFrame):
        index_name = failure_cases.index.name
        failure_cases = (
            failure_cases
            .pipe(lambda df: pd.Series(
                df.itertuples()).map(lambda x: x.__repr__()))
            .rename("failure_case")
            .reset_index()
        )
    elif isinstance(failure_cases, pd.Series):
        index_name = failure_cases.index.name
        failure_cases = (
            failure_cases
            .rename("failure_case")
            .reset_index()
        )
    else:
        raise TypeError(
            "type of failure_cases argument not understood: %s" %
            type(failure_cases))

    index_name = "index" if index_name is None else index_name
    failure_cases = (
        failure_cases
        .groupby("failure_case")[index_name].agg([list, len])
        .rename(columns={"list": index_name, "len": "count"})
        .sort_values("count", ascending=False)
    )

    return failure_cases.head(n_cases)
