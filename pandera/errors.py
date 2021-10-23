"""pandera-specific errors."""

from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Union

import pandas as pd

ErrorData = namedtuple(
    "ErrorData",
    [
        "data",
        "error_counts",
        "column_errors",
        "type_errors",
        "check_errors",
    ],
)


class ParserError(Exception):
    """Raised when data cannot be parsed from the raw into its clean form."""

    def __init__(self, message, failure_cases):
        super().__init__(message)
        self.failure_cases = failure_cases


class SchemaInitError(Exception):
    """Raised when schema initialization fails."""


class SchemaDefinitionError(Exception):
    """Raised when schema definition is invalid on object validation."""


class SchemaError(Exception):
    """Raised when object does not pass schema validation constraints."""

    def __init__(
        self,
        schema,
        data,
        message,
        failure_cases=None,
        check=None,
        check_index=None,
        check_output=None,
    ):
        super().__init__(message)
        self.schema = schema
        self.data = data
        self.failure_cases = failure_cases
        self.check = check
        self.check_index = check_index
        self.check_output = check_output


class BaseStrategyOnlyError(Exception):
    """Custom error for reporting strategies that must be base strategies."""


SCHEMA_ERRORS_SUFFIX = """

Usage Tip
---------

Directly inspect all errors by catching the exception:

```
try:
    schema.validate(dataframe, lazy=True)
except SchemaErrors as err:
    err.failure_cases  # dataframe of schema errors
    err.data  # invalid dataframe
```
"""


class SchemaErrors(Exception):
    """Raised when multiple schema are lazily collected into one error."""

    def __init__(
        self,
        schema_errors: List[Dict[str, Any]],
        data: Union[pd.Series, pd.DataFrame],
    ):
        error_counts, failure_cases = self._parse_schema_errors(schema_errors)
        super().__init__(self._message(error_counts, failure_cases))
        self.schema_errors = schema_errors
        self.error_counts = error_counts
        self.failure_cases = failure_cases
        self.data = data

    @staticmethod
    def _message(error_counts, schema_errors):
        """Format error message."""
        msg = (
            f"A total of {sum(error_counts.values())} "
            "schema errors were found.\n"
        )

        msg += "\nError Counts"
        msg += "\n------------\n"
        for k, v in error_counts.items():
            msg += f"- {k}: {v}\n"

        def agg_failure_cases(df):
            # NOTE: this is a hack to add modin support
            if type(df).__module__.startswith("modin.pandas"):
                return (
                    df.groupby(["schema_context", "column", "check"])
                    .agg({"failure_case": "unique"})
                    .failure_case
                )
            return df.groupby(
                ["schema_context", "column", "check"]
            ).failure_case.unique()

        agg_schema_errors = (
            schema_errors.fillna({"column": "<NA>"})
            .pipe(agg_failure_cases)
            .rename("failure_cases")
            .to_frame()
            .assign(n_failure_cases=lambda df: df.failure_cases.map(len))
        )
        index_labels = [
            agg_schema_errors.index.names.index(name)
            for name in ["schema_context", "column"]
        ]
        agg_schema_errors = agg_schema_errors.sort_index(
            level=index_labels,
            ascending=[False, True],
        )
        msg += "\nSchema Error Summary"
        msg += "\n--------------------\n"
        with pd.option_context("display.max_colwidth", 100):
            msg += agg_schema_errors.to_string()
        msg += SCHEMA_ERRORS_SUFFIX
        return msg

    @staticmethod
    def _parse_schema_errors(schema_errors: List[Dict[str, Any]]):
        """Parse schema error dicts to produce data for error message."""
        error_counts = defaultdict(int)  # type: ignore
        check_failure_cases = []

        column_order = [
            "schema_context",
            "column",
            "check",
            "check_number",
            "failure_case",
            "index",
        ]

        for schema_error_dict in schema_errors:
            reason_code = schema_error_dict["reason_code"]
            err = schema_error_dict["error"]

            error_counts[reason_code] += 1
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
                        if reason_code == "schema_component_check"
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

        # NOTE: this is a hack to support koalas and modin
        concat_fn = pd.concat
        if any(
            type(x).__module__.startswith("databricks.koalas")
            for x in check_failure_cases
        ):
            # pylint: disable=import-outside-toplevel
            import databricks.koalas as ks

            concat_fn = ks.concat
            check_failure_cases = [
                x if isinstance(x, ks.DataFrame) else ks.DataFrame(x)
                for x in check_failure_cases
            ]
        elif any(
            type(x).__module__.startswith("modin.pandas")
            for x in check_failure_cases
        ):
            # pylint: disable=import-outside-toplevel
            import modin.pandas as mpd

            concat_fn = mpd.concat
            check_failure_cases = [
                x if isinstance(x, mpd.DataFrame) else mpd.DataFrame(x)
                for x in check_failure_cases
            ]

        failure_cases = (
            concat_fn(check_failure_cases)
            .reset_index(drop=True)
            .sort_values("schema_context", ascending=False)
            .drop_duplicates()
        )
        return error_counts, failure_cases
