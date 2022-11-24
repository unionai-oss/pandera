"""pandera-specific errors."""

import warnings
from enum import Enum
from typing import Any, Dict, List, NamedTuple


class ReducedPickleExceptionBase(Exception):
    """Base class for Excption with non-conserved state under pickling.

    Derived classes define attributes to be transformed to
    string via `TO_STRING_KEYS`.
    """

    TO_STRING_KEYS: List[str] = []

    def __reduce__(self):
        """Exception.__reduce__ is incompatible. Override with custom layout.

        Each attribute in `TO_STRING_KEYS` is replaced by its string
        representation.
        """
        state = {
            key: str(val)
            if key in self.TO_STRING_KEYS and val is not None
            else val
            for key, val in self.__dict__.items()
        }
        state["args"] = self.args  # message may not be in __dict__
        return (
            self.__class__.__new__,  # object creation function
            (self.__class__,),  # arguments to said function
            state,  # arguments to `__setstate__` after creation
        )

    @classmethod
    def _unpickle_warning(cls):
        """Create the warning message about state loss in pickling."""
        return (
            f"Pickling {cls.__name__} does not preserve state: "
            f"Attributes {cls.TO_STRING_KEYS} become string "
            "representations."
        )

    def __setstate__(self, state):
        """Show warning during unpickling."""
        warnings.warn(self._unpickle_warning())
        return super().__setstate__(state)


class ParserError(ReducedPickleExceptionBase):
    """Raised when data cannot be parsed from the raw into its clean form."""

    TO_STRING_KEYS = ["failure_cases"]

    def __init__(self, message, failure_cases):
        super().__init__(message)
        self.failure_cases = failure_cases


class SchemaInitError(Exception):
    """Raised when schema initialization fails."""


class SchemaDefinitionError(Exception):
    """Raised when schema definition is invalid on object validation."""


class SchemaError(ReducedPickleExceptionBase):
    """Raised when object does not pass schema validation constraints."""

    TO_STRING_KEYS = [
        "schema",
        "data",
        "failure_cases",
        "check",
        "check_output",
        "reason_code",
    ]

    def __init__(
        self,
        schema,
        data,
        message,
        failure_cases=None,
        check=None,
        check_index=None,
        check_output=None,
        reason_code=None,
    ):
        super().__init__(message)
        self.schema = schema
        self.data = data
        self.failure_cases = failure_cases
        self.check = check
        self.check_index = check_index
        self.check_output = check_output
        self.reason_code = reason_code


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


class FailureCaseMetadata(NamedTuple):
    """Consolidated failure cases, summary message, and error counts."""

    failure_cases: Any
    message: str
    error_counts: Dict[str, int]


class SchemaErrorReason(Enum):
    """Reason codes for schema errors."""

    INVALID_TYPE = "invalid_type"
    DATATYPE_COERCION = "dtype_coercion_error"
    COLUMN_NOT_IN_SCHEMA = "column_not_in_schema"
    COLUMN_NOT_ORDERED = "column_not_ordered"
    DUPLICATE_COLUMN_LABELS = "duplicate_dataframe_column_labels"
    COLUMN_NOT_IN_DATAFRAME = "column_not_in_dataframe"
    SCHEMA_COMPONENT_CHECK = "schema_component_check"
    DATAFRAME_CHECK = "dataframe_check"
    CHECK_ERROR = "check_error"
    DUPLICATES = "duplicates"
    WRONG_FIELD_NAME = "wrong_field_name"
    SERIES_CONTAINS_NULLS = "series_contains_nulls"
    SERIES_CONTAINS_DUPLICATES = "series_contains_duplicates"
    SERIES_CHECK = "series_check"
    WRONG_DATATYPE = "wrong_dtype"
    INDEX_CHECK = "index_check"


class SchemaErrors(ReducedPickleExceptionBase):
    """Raised when multiple schema are lazily collected into one error."""

    TO_STRING_KEYS = [
        "schema",
        "failure_cases",
        "data",
    ]

    def __init__(
        self,
        schema,
        schema_errors: List[Dict[str, Any]],
        data: Any,
    ):
        self.schema = schema
        self.schema_errors = schema_errors
        self.data = data

        failure_cases_metadata = schema.BACKEND.failure_cases_metadata(
            schema.name, schema_errors
        )
<<<<<<< HEAD
        self.error_counts = failure_cases_metadata.error_counts
        self.failure_cases = failure_cases_metadata.failure_cases
        super().__init__(failure_cases_metadata.message)
=======

        msg += "\nError Counts"
        msg += "\n------------\n"
        for k, v in error_counts.items():
            msg += f"- {k}: {v}\n"

        def agg_failure_cases(df):
            # Note: hack to support unhashable types, proper solution that only transforms
            # when requires https://github.com/unionai-oss/pandera/issues/260
            df.failure_case = df.failure_case.astype(str)
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
                        if reason_code
                        == SchemaErrorReason.SCHEMA_COMPONENT_CHECK.value
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
        concat_fn = pd.concat
        if any(
            type(x).__module__.startswith("pyspark.pandas")
            for x in check_failure_cases
        ):
            # pylint: disable=import-outside-toplevel
            import pyspark.pandas as ps

            concat_fn = ps.concat  # type: ignore[assignment]
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

            concat_fn = mpd.concat
            check_failure_cases = [
                x if isinstance(x, mpd.DataFrame) else mpd.DataFrame(x)
                for x in check_failure_cases
            ]

        failure_cases = (
            concat_fn(check_failure_cases)
            .reset_index(drop=True)
            .sort_values("schema_context", ascending=False)
        )
        return error_counts, failure_cases
>>>>>>> 2235fab (chore: formatting and docs)
