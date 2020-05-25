"""Make schema error messages human-friendly."""

from typing import Union

import pandas as pd

from .checks import Check
from .hypotheses import Hypothesis


def format_generic_error_message(
        parent_schema,
        check: Union[Check, Hypothesis],
        check_index: int,
) -> str:
    """Construct an error message when a check validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    """
    return "%s failed series validator %d:\n%s" % \
        (parent_schema, check_index, check)


def format_vectorized_error_message(
        parent_schema,
        check: Union[Check, Hypothesis],
        check_index: int,
        reshaped_failure_cases: pd.DataFrame) -> str:
    """Construct an error message when a validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    :param reshaped_failure_cases: The failure cases encountered by the
        element-wise or vectorized validator.

    """
    agg_failure_cases = (
        reshaped_failure_cases
        .groupby("failure_case")["index"].agg([list, len])
        .rename(columns={"list": "index", "len": "count"})
        .sort_values("count", ascending=False)
        .head(check.n_failure_cases)
    )
    return (
        "%s failed element-wise validator %d:\n"
        "%s\nfailure cases:\n%s" % (
            parent_schema,
            check_index,
            check,
            agg_failure_cases,
        )
    )


def scalar_failure_case(x) -> pd.DataFrame:
    """Construct failure case from a scalar value.

    :param x: a scalar value representing failure case.
    :returns: DataFrame used for error reporting with ``SchemaErrors``.
    """
    return pd.DataFrame({
        "index": [None],
        "failure_case": [x],
    })


def reshape_failure_cases(
        failure_cases: Union[pd.DataFrame, pd.Series],
        ignore_na: bool = True
) -> pd.DataFrame:
    """Construct readable error messages for vectorized_error_message.

    :param failure_cases: The failure cases encountered by the element-wise
        or vectorized validator.
    :param ignore_na: whether or not to ignore null failure cases.
    :returns: DataFrame where index contains failure cases, the "index"
        column contains a list of integer indexes in the validation
        DataFrame that caused the failure, and a "count" column
        representing how many failures of that case occurred.

    """
    if hasattr(failure_cases, "index") and \
            isinstance(failure_cases.index, pd.MultiIndex):
        failure_cases = (
            failure_cases
            .rename("failure_case")
            .reset_index()
            .assign(
                index=lambda df: (
                    df.apply(tuple, axis=1).astype(str)
                )
            )[["failure_case", "index"]]
        )
    elif isinstance(failure_cases, pd.DataFrame):
        failure_cases = (
            failure_cases
            .rename_axis("column", axis=1)
            .rename_axis("index", axis=0)
            .unstack()
            .rename("failure_case")
            .reset_index()
        )
    elif isinstance(failure_cases, pd.Series):
        failure_cases = (
            failure_cases
            .rename("failure_case")
            .rename_axis("index")
            .reset_index()
        )
    else:
        raise TypeError(
            "type of failure_cases argument not understood: %s" %
            type(failure_cases))

    return failure_cases.dropna() if ignore_na else failure_cases
