"""Backend implementation for pyspark schema components."""

import traceback
from copy import copy, deepcopy
from typing import Iterable, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import cast

from pandera.backends.pyspark.column import ColumnSchemaBackend
from pandera.backends.pyspark.container import DataFrameSchemaBackend
from pandera.api.pyspark.types import (
    # is_field,
    # is_index, # Don't need this
    # is_multiindex, # Don't need this
    is_table,
)
from pandera.backends.pandas.error_formatters import scalar_failure_case
from pandera.api.pyspark.error_handler import ErrorHandler, ErrorCategory
from pandera.errors import SchemaError, SchemaErrors, SchemaErrorReason
import re


class ColumnBackend(ColumnSchemaBackend):
    """Backend implementation for pyspark dataframe columns."""

    def report_errors(
        self,
        check_obj: DataFrame,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        error_handler: ErrorHandler,
    ) -> DataFrame:
        """Validation backend implementation for pyspark dataframe columns.."""
        # error_handler = ErrorHandler(lazy)
        if schema.name is None:
            raise SchemaError(
                schema,
                check_obj,
                "column name is set to None. Pass the ``name` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.",
            )

        def validate_column(check_obj, column_name):
            try:
                # pylint: disable=super-with-arguments
                super(ColumnBackend, self).report_errors(
                    check_obj,
                    copy(schema).set_name(column_name),
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=inplace,
                    error_handler=error_handler,
                )
            # except SchemaErrors as err:
            #     for err_dict in err.schema_errors:
            #
            #         error_handler.collect_error(
            #             ErrorCategory.DATA, err_dict["reason_code"], err_dict["error"]
            #         )
            except SchemaError as err:
                error_handler.collect_error(ErrorCategory.DATA, err.reason_code, err)

        column_keys_to_check = (
            self.get_regex_columns(schema, check_obj.columns, check_obj)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            if schema.coerce:
                check_obj = self.coerce_dtype(
                    check_obj,
                    schema=schema,
                    error_handler=error_handler,
                )
            validate_column(check_obj, column_name)

            # if is_table(check_obj[column_name]):
            #     for i in range(check_obj[column_name].shape[1]):
            #         validate_column(
            #             check_obj[column_name].iloc[:, [i]], column_name
            #         )
            # else:
            #    validate_column(check_obj, column_name)

        # if lazy and error_handler.collected_errors:
        #     raise SchemaErrors(
        #         schema=schema,
        #         schema_errors=error_handler.collected_errors,
        #         data=check_obj,
        #     )

        return check_obj

    def get_regex_columns(self, schema, columns) -> Iterable:
        """Get matching column names based on regex column name pattern.

        :param schema: schema specification to use
        :param columns: columns to regex pattern match
        :returns: matchin columns
        """
        pattern = re.compile(schema.name)
        column_keys_to_check = [
            col_name for col_name in columns if pattern.match(col_name)
        ]
        if len(column_keys_to_check) == 0:
            raise SchemaError(
                schema=schema,
                data=columns,
                message=(
                    f"Column regex name='{schema.name}' did not match any "
                    "columns in the dataframe. Update the regex pattern so "
                    f"that it matches at least one column:\n{columns.tolist()}",
                ),
                failure_cases=scalar_failure_case(str(columns)),
                check=f"no_regex_column_match('{schema.name}')",
            )
        # drop duplicates to account for potential duplicated columns in the
        # dataframe.
        return column_keys_to_check

    def coerce_dtype(
        self,
        check_obj: DataFrame,
        *,
        schema=None,
    ) -> DataFrame:
        """Coerce dtype of a column, handling duplicate column names."""
        # pylint: disable=super-with-arguments
        # pylint: disable=fixme
        # TODO: use singledispatchmethod here

        check_obj = check_obj.withColumn(schema.name, cast(schema.dtype))
        # return check_obj.apply(
        #     lambda x: super(ColumnBackend, self).coerce_dtype(
        #         x,
        #         schema=schema,
        #         error_handler=error_handler,
        #     ),
        #     axis="columns",
        # )
        return check_obj

    def run_checks(self, check_obj, schema, error_handler, lazy):
        check_results = []
        for check_index, check in enumerate(schema.checks):
            check_args = [schema.name]
            try:
                check_results.append(
                    self.run_check(check_obj, schema, check, check_index, *check_args)
                )
            except SchemaError as err:
                error_handler.collect_error(
                    type=ErrorCategory.DATA,
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                    schema_error=err,
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"

                error_handler.collect_error(
                    type=ErrorCategory.DATA,
                    reason_code=SchemaErrorReason.CHECK_ERROR,
                    schema_error=SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=(
                            f"Error while executing check function: {err_str}\n"
                            + traceback.format_exc()
                        ),
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )
        return check_results
