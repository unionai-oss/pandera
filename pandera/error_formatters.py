"""Make schema error messages human-friendly."""

from typing import Union

import pandas as pd

from . import check_utils


def format_generic_error_message(
    parent_schema,
    check,
    check_index: int,
) -> str:
    """Construct an error message when a check validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    """
    return (
        f"{parent_schema} failed series or dataframe validator "
        f"{check_index}:\n{check}"
    )


def format_vectorized_error_message(
    parent_schema,
    check,
    check_index: int,
    reshaped_failure_cases: pd.DataFrame,
) -> str:
    """Construct an error message when a validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    :param reshaped_failure_cases: The failure cases encountered by the
        element-wise or vectorized validator.

    """
    return (
        f"{parent_schema} failed element-wise validator {check_index}:\n"
        f"{check}\nfailure cases:\n{reshaped_failure_cases}"
    )


def scalar_failure_case(x) -> pd.DataFrame:
    """Construct failure case from a scalar value.

    :param x: a scalar value representing failure case.
    :returns: DataFrame used for error reporting with ``SchemaErrors``.
    """
    return pd.DataFrame(
        {
            "index": [None],
            "failure_case": [x],
        }
    )


def reshape_failure_cases(
    failure_cases: Union[pd.DataFrame, pd.Series], ignore_na: bool = True
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
    if not (
        check_utils.is_table(failure_cases)
        or check_utils.is_field(failure_cases)
    ):
        raise TypeError(
            "Expected failure_cases to be a DataFrame or Series, found "
            f"{type(failure_cases)}"
        )

    if (
        check_utils.is_table(failure_cases)
        and "column" in failure_cases.columns
        and "failure_case" in failure_cases.columns
    ):
        reshaped_failure_cases = failure_cases
    elif check_utils.is_table(failure_cases) and check_utils.is_multiindex(
        failure_cases.index
    ):
        reshaped_failure_cases = (
            failure_cases.rename_axis("column", axis=1)
            .assign(
                index=lambda df: (
                    df.index.to_frame().apply(tuple, axis=1).astype(str)
                )
            )
            .set_index("index", drop=True)
            .unstack()
            .rename("failure_case")
            .reset_index()
        )
    elif check_utils.is_field(failure_cases) and check_utils.is_multiindex(
        failure_cases.index
    ):
        reshaped_failure_cases = (
            failure_cases.rename("failure_case")
            .to_frame()
            .assign(
                index=lambda df: (
                    df.index.to_frame().apply(tuple, axis=1).astype(str)
                )
            )[["failure_case", "index"]]
            .reset_index(drop=True)
        )
    elif check_utils.is_table(failure_cases):
        reshaped_failure_cases = failure_cases.unstack().reset_index()
        reshaped_failure_cases.columns = ["column", "index", "failure_case"]
    elif check_utils.is_field(failure_cases):
        reshaped_failure_cases = failure_cases.rename("failure_case")
        reshaped_failure_cases.index.name = "index"
        reshaped_failure_cases = reshaped_failure_cases.reset_index()
    else:
        raise TypeError(
            "type of failure_cases argument not understood: "
            f"{type(failure_cases)}"
        )

    return (
        reshaped_failure_cases.dropna()
        if ignore_na
        else reshaped_failure_cases
    )
