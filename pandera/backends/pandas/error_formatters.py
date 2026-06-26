"""Make schema error messages human-friendly."""

import re
from typing import Any, Union

import pandas as pd

from pandera.errors import SchemaError

# Cap how many categories are listed in a dtype-mismatch message; mirrors the
# default ``n_failure_cases`` limit so high-cardinality categoricals don't
# produce multi-KB error strings.
_MAX_CATEGORIES_IN_MESSAGE = 10


def describe_dtype_mismatch(expected_dtype, actual_dtype) -> tuple[str, str]:
    """Describe an expected/actual dtype pair for a dtype-mismatch message.

    Categorical dtypes all stringify to the bare word ``category``, so a
    mismatch between two categoricals reads ``category, got category`` and names
    nothing (see issue #2051). When both sides stringify identically and expose
    ``.categories``, surface the categories on each side; when those match too,
    surface the differing ``ordered`` flag instead. Non-colliding mismatches are
    returned unchanged.
    """
    expected_str, actual_str = str(expected_dtype), str(actual_dtype)
    if expected_str != actual_str:
        return expected_str, actual_str

    expected_categories = getattr(expected_dtype, "categories", None)
    actual_categories = getattr(actual_dtype, "categories", None)
    if expected_categories is None or actual_categories is None:
        return expected_str, actual_str

    def _describe(dtype, categories) -> str:
        categories = tuple(categories)
        if len(categories) > _MAX_CATEGORIES_IN_MESSAGE:
            shown = ", ".join(
                repr(c) for c in categories[:_MAX_CATEGORIES_IN_MESSAGE]
            )
            remaining = len(categories) - _MAX_CATEGORIES_IN_MESSAGE
            return (
                f"{dtype} with {len(categories)} categories "
                f"({shown}, ... +{remaining} more)"
            )
        return f"{dtype} with categories {categories}"

    expected_desc = _describe(expected_dtype, expected_categories)
    actual_desc = _describe(actual_dtype, actual_categories)
    if tuple(expected_categories) == tuple(actual_categories):
        # Categories match on both sides, so the mismatch is the ``ordered``
        # flag; surface it so the two descriptions differ.
        expected_desc += (
            f", ordered={getattr(expected_dtype, 'ordered', None)}"
        )
        actual_desc += f", ordered={getattr(actual_dtype, 'ordered', None)}"
    return expected_desc, actual_desc


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
        f"{parent_schema.__class__.__name__} '{parent_schema.name}' failed "
        "series or dataframe validator "
        f"{check_index}: {check}"
    )


def format_vectorized_error_message(
    parent_schema,
    check,
    check_index: int,
    reshaped_failure_cases: Any,
) -> str:
    """Construct an error message when a validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    :param reshaped_failure_cases: The failure cases encountered by the
        element-wise or vectorized validator.

    """

    pattern = r"<Check\s+([^:>]+):\s*([^>]+)>"
    matches = re.findall(pattern, str(check))

    check_strs = [f"{match[1]}" for match in matches]

    if check_strs:
        check_str = check_strs[0]
    else:
        check_str = str(check)

    if type(reshaped_failure_cases.failure_case).__module__.startswith(
        "pyspark.pandas"
    ):
        failure_cases = reshaped_failure_cases.failure_case.to_numpy()
        failure_cases_string = ", ".join(failure_cases.astype(str))
    else:
        failure_cases = reshaped_failure_cases.failure_case
        failure_cases_string = ", ".join(failure_cases.apply(str))

    return (
        f"{parent_schema.__class__.__name__} '{parent_schema.name}' failed "
        f"element-wise validator number {check_index}: "
        f"{check_str} failure cases: {failure_cases_string}"
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

    from pandera.api.pandas.types import is_field, is_multiindex, is_table

    if not (is_table(failure_cases) or is_field(failure_cases)):
        raise TypeError(
            "Expected failure_cases to be a DataFrame or Series, found "
            f"{type(failure_cases)}"
        )

    if (
        is_table(failure_cases)
        and "column" in failure_cases.columns
        and "failure_case" in failure_cases.columns
    ):
        reshaped_failure_cases = failure_cases
    elif is_table(failure_cases) and is_multiindex(failure_cases.index):
        reshaped_failure_cases = (
            failure_cases.rename_axis("column", axis=1)  # type: ignore[call-overload]
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
    elif is_field(failure_cases) and is_multiindex(failure_cases.index):
        reshaped_failure_cases = (
            failure_cases.rename("failure_case")  # type: ignore[call-overload]
            .to_frame()
            .assign(
                index=lambda df: (
                    _multiindex_to_frame(df).apply(tuple, axis=1).astype(str)
                )
            )[["failure_case", "index"]]
            .reset_index(drop=True)
        )
    elif is_table(failure_cases):
        reshaped_failure_cases = failure_cases.unstack().reset_index()
        reshaped_failure_cases.columns = ["column", "index", "failure_case"]  # type: ignore[call-overload,assignment]  # noqa
    elif is_field(failure_cases):
        reshaped_failure_cases = failure_cases.rename("failure_case")  # type: ignore[call-overload]
        reshaped_failure_cases.index.name = "index"
        reshaped_failure_cases = reshaped_failure_cases.reset_index()
    else:
        raise TypeError(
            "type of failure_cases argument not understood: "
            f"{type(failure_cases)}"
        )

    return (
        reshaped_failure_cases.dropna()  # type: ignore[return-value]
        if ignore_na
        else reshaped_failure_cases
    )


def _multiindex_to_frame(df):
    from pandera.engines.utils import pandas_version

    if pandas_version().release >= (1, 5, 0):
        return df.index.to_frame(allow_duplicates=True)
    return df.index.to_frame().drop_duplicates()


def consolidate_failure_cases(schema_errors: list[SchemaError]):
    """Consolidate schema error dicts to produce data for error message."""
    from pandera.api.pandas.types import is_table

    assert schema_errors, (
        "schema_errors input cannot be empty. Check how the backend "
        "validation logic is handling/raising SchemaError(s)."
    )
    check_failure_cases = []
    scalar_check_failure_cases = []

    column_order = [
        "schema_context",
        "column",
        "check",
        "check_number",
        "failure_case",
        "index",
    ]

    for err in schema_errors:
        check_identifier = (
            None
            if err.check is None
            else (
                err.check
                if isinstance(err.check, str)
                else (
                    err.check.error
                    if err.check.error is not None
                    else (
                        err.check.name
                        if err.check.name is not None
                        else str(err.check)
                    )
                )
            )
        )

        err_metadata = {
            "schema_context": err.schema.__class__.__name__,
            "check": check_identifier,
            "check_number": err.check_index,
        }

        if is_table(err.failure_cases):
            if "column" in err.failure_cases:
                column = err.failure_cases["column"]
            elif err.column_name is not None:
                column = err.column_name
            else:
                column = err.schema.name

            failure_cases = err.failure_cases.assign(
                column=(
                    # if the column key is a tuple (for MultiIndex column
                    # names), explicitly wrap `column` in a list of the
                    # same length as the number of failure cases.
                    [column] * err.failure_cases.shape[0]
                    if isinstance(column, tuple)
                    else column
                ),
                **err_metadata,
            )
            check_failure_cases.append(failure_cases[column_order])
        else:
            # assume that err.failure_cases is a scalar value
            scalar_check_failure_cases.append(
                {
                    "column": err.schema.name,
                    "failure_case": err.failure_cases,
                    "index": None,
                    **err_metadata,
                }
            )

    if scalar_check_failure_cases:
        check_failure_cases.append(
            pd.DataFrame.from_records(scalar_check_failure_cases)
        )

    # NOTE: this is a hack to support pyspark.pandas and modin
    concat_fn = pd.concat  # type: ignore
    if any(
        type(x).__module__.startswith("pyspark.pandas")
        for x in check_failure_cases
    ):
        import pyspark.pandas as ps

        concat_fn = ps.concat  # type: ignore
        check_failure_cases = [
            x if isinstance(x, ps.DataFrame) else ps.DataFrame(x)
            for x in check_failure_cases
        ]
        return (
            concat_fn(check_failure_cases)
            .reset_index(drop=True)
            .sort_values("schema_context", ascending=False)
        )
    elif any(
        type(x).__module__.startswith("modin.pandas")
        for x in check_failure_cases
    ):
        import modin.pandas as mpd

        concat_fn = mpd.concat  # type: ignore
        check_failure_cases = [
            x if isinstance(x, mpd.DataFrame) else mpd.DataFrame(x)
            for x in check_failure_cases
        ]
        return concat_fn(check_failure_cases).reset_index(drop=True)

    return (
        concat_fn(check_failure_cases)
        .reset_index(drop=True)
        .sort_values("schema_context", ascending=False)
    )
