"""Make schema error messages human-friendly."""
import textwrap

from collections import defaultdict
from typing import Dict, List, Tuple, Union
import pandas as pd

from pandera.errors import SchemaError, SchemaErrorReason


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
    # pylint: disable=import-outside-toplevel,cyclic-import
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
    # pylint: disable=import-outside-toplevel,cyclic-import
    from pandera.engines.utils import pandas_version

    if pandas_version().release >= (1, 5, 0):
        return df.index.to_frame(allow_duplicates=True)
    return df.index.to_frame().drop_duplicates()


def consolidate_failure_cases(schema_errors: List[SchemaError]):
    """Consolidate schema error dicts to produce data for error message."""
    assert schema_errors, (
        "schema_errors input cannot be empty. Check how the backend "
        "validation logic is handling/raising SchemaError(s)."
    )
    check_failure_cases = []

    column_order = [
        "schema_context",
        "column",
        "check",
        "check_number",
        "failure_case",
        "index",
    ]

    for schema_error in schema_errors:
        err, reason_code = schema_error, schema_error.reason_code

        check_identifier = (
            None
            if err.check is None
            else err.check
            if isinstance(err.check, str)
            else err.check.error
            if err.check.error is not None
            else err.check.name
            if err.check.name is not None
            else str(err.check)
        )

        if err.failure_cases is not None:
            if "column" in err.failure_cases:
                column = err.failure_cases["column"]
            else:
                column = (
                    err.schema.name
                    if reason_code
                    in {
                        SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        SchemaErrorReason.DATAFRAME_CHECK,
                    }
                    else None
                )

            failure_cases = err.failure_cases.assign(
                schema_context=err.schema.__class__.__name__,
                check=check_identifier,
                check_number=err.check_index,
                # if the column key is a tuple (for MultiIndex column
                # names), explicitly wrap `column` in a list of the
                # same length as the number of failure cases.
                column=(
                    [column] * err.failure_cases.shape[0]
                    if isinstance(column, tuple)
                    else column
                ),
            )
            check_failure_cases.append(failure_cases[column_order])

    # NOTE: this is a hack to support pyspark.pandas and modin
    concat_fn = pd.concat  # type: ignore
    if any(
        type(x).__module__.startswith("pyspark.pandas")
        for x in check_failure_cases
    ):
        # pylint: disable=import-outside-toplevel
        import pyspark.pandas as ps

        concat_fn = ps.concat  # type: ignore
        check_failure_cases = [
            x if isinstance(x, ps.DataFrame) else ps.DataFrame(x)
            for x in check_failure_cases
        ]
    elif any(
        type(x).__module__.startswith("modin.pandas")
        for x in check_failure_cases
    ):
        # pylint: disable=import-outside-toplevel
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


def summarize_failure_cases(
    schema_name: str,
    schema_errors: List[SchemaError],
    failure_cases: pd.DataFrame,
) -> Tuple[str, Dict[str, int]]:
    """Format error message."""

    if schema_name is None:
        schema_name = "Nameless Schema"

    msg = ""
    msg += "\nSchema Error Summary"
    msg += "\n--------------------\n"
    msg = (
        f"Schema {schema_name}: A total of "
        f"{len(failure_cases)} schema errors were found.\n"
    )

    failure_cases_table = dataframe_to_tabular_formatted_string(
        failure_cases,
        column_order=[
            "index",
            "column",
            "check",
            "failure_case",
            "schema_context",
            "check_number",
        ],
    )
    msg += failure_cases_table

    return msg, len(schema_errors)


def dataframe_to_tabular_formatted_string(
    dataframe, limit=10, max_width=20, max_cell_length=200, column_order=None
):
    formatted_string = ""

    # Reorder DataFrame columns if column_order is provided
    if column_order:
        dataframe = dataframe[column_order]

    limited_df = dataframe.head(limit)

    # Truncate cell contents to max_cell_length and then determine the maximum width for each column
    truncated_df = limited_df.applymap(
        lambda x: str(x)[:max_cell_length] if isinstance(x, str) else x
    )
    column_widths = {
        column: min(
            max(len(column), truncated_df[column].astype(str).map(len).max()),
            max_width,
        )
        for column in truncated_df.columns
    }

    # Top border
    formatted_string += (
        "┌"
        + "┬".join(["─" * (width + 2) for width in column_widths.values()])
        + "┐\n"
    )

    # Headers
    formatted_string += (
        "│ "
        + " ┆ ".join(
            [column.center(width) for column, width in column_widths.items()]
        )
        + " │\n"
    )

    # Separator
    formatted_string += (
        "╞"
        + "╪".join(["═" * (width + 2) for width in column_widths.values()])
        + "╡\n"
    )

    # Each row
    for _, row in truncated_df.iterrows():
        wrapped_row = [
            [line for line in textwrap.wrap(str(value), width)]
            for value, width in zip(row, column_widths.values())
        ]
        max_lines = max(len(wrapped) for wrapped in wrapped_row)

        for line in range(max_lines):
            line_data = " │ ".join(
                (
                    wrapped_row[i][line] if line < len(wrapped_row[i]) else ""
                ).ljust(column_widths[truncated_df.columns[i]])
                for i in range(len(truncated_df.columns))
            )
            formatted_string += "│ " + line_data + " │\n"

    # Bottom border and additional rows message if applicable
    if len(dataframe) > limit:
        formatted_string += (
            "│ " + " ┆ ".join(["…" for _ in column_widths.values()]) + " │\n"
        )
        formatted_string += f"And {len(dataframe) - limit} others"
    else:
        formatted_string += (
            "└"
            + "┴".join(["─" * (width + 2) for width in column_widths.values()])
            + "┘\n"
        )

    return formatted_string
