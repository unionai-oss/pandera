"""Backend implementation for pandas schema components."""

import traceback
from copy import copy, deepcopy
from typing import Iterable, Optional, Union

from pyspark.sql import DataFrame

from pandera.backends.pyspark.array import ArraySchemaBackend
from pandera.backends.pyspark.container import DataFrameSchemaBackend
from pandera.api.pyspark.types import (
    is_field,
    #is_index, # Don't need this
    #is_multiindex, # Don't need this
    is_table,
)
from pandera.backends.pandas.error_formatters import scalar_failure_case
from pandera.error_handlers import SchemaErrorHandler
from pandera.errors import SchemaError, SchemaErrors, SchemaErrorReason


class ColumnBackend(ArraySchemaBackend):
    """Backend implementation for pandas dataframe columns."""

    def validate(
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
    ) -> DataFrame:
        """Validation backend implementation for pandas dataframe columns.."""
        error_handler = SchemaErrorHandler(lazy=lazy)

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
                super(ColumnBackend, self).validate(
                    check_obj,
                    copy(schema).set_name(column_name),
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=inplace,
                )
            except SchemaErrors as err:
                for err_dict in err.schema_errors:
                    error_handler.collect_error(
                        err_dict["reason_code"], err_dict["error"]
                    )
            except SchemaError as err:
                error_handler.collect_error(err.reason_code, err)

        column_keys_to_check = (
            self.get_regex_columns(schema, check_obj.columns)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            if schema.coerce:
                check_obj[column_name] = self.coerce_dtype(
                    check_obj[column_name],
                    schema=schema,
                    error_handler=error_handler,
                )

            if is_table(check_obj[column_name]):
                for i in range(check_obj[column_name].shape[1]):
                    validate_column(
                        check_obj[column_name].iloc[:, [i]], column_name
                    )
            else:
                validate_column(check_obj, column_name)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.collected_errors,
                data=check_obj,
            )

        return check_obj

    def get_regex_columns(
        self, schema, columns
    ) -> Iterable:
        """Get matching column names based on regex column name pattern.

        :param schema: schema specification to use
        :param columns: columns to regex pattern match
        :returns: matchin columns
        """

        column_keys_to_check = columns[
            # str.match will return nan values when the index value is
            # not a string.
            pd.Index(columns.astype(str).str.match(schema.name))
            .fillna(False)
            .tolist()
        ]
        if column_keys_to_check.shape[0] == 0:
            raise SchemaError(
                schema=schema,
                data=columns,
                message=(
                    f"Column regex name='{schema.name}' did not match any "
                    "columns in the dataframe. Update the regex pattern so "
                    f"that it matches at least one column:\n{columns.tolist()}",
                ),
                failure_cases=scalar_failure_case(str(columns.tolist())),
                check=f"no_regex_column_match('{schema.name}')",
            )
        # drop duplicates to account for potential duplicated columns in the
        # dataframe.
        return column_keys_to_check.drop_duplicates()

    def coerce_dtype(
        self,
        check_obj: Union[DataFrame, col],
        *,
        schema=None,
        error_handler: SchemaErrorHandler = None,
    ) -> Union[DataFrame, col]:
        """Coerce dtype of a column, handling duplicate column names."""
        # pylint: disable=super-with-arguments
        # pylint: disable=fixme
        # TODO: use singledispatchmethod here
        if is_field(check_obj):
            return super(ColumnBackend, self).coerce_dtype(
                check_obj,
                schema=schema,
                error_handler=error_handler,
            )
        return check_obj.apply(
            lambda x: super(ColumnBackend, self).coerce_dtype(
                x,
                schema=schema,
                error_handler=error_handler,
            ),
            axis="columns",
        )

    def run_checks(self, check_obj, schema, error_handler, lazy):
        check_results = []
        for check_index, check in enumerate(schema.checks):
            check_args = [None] if is_field(check_obj) else [schema.name]
            try:
                check_results.append(
                    self.run_check(
                        check_obj, schema, check, check_index, *check_args
                    )
                )
            except SchemaError as err:
                error_handler.collect_error(
                    SchemaErrorReason.DATAFRAME_CHECK,
                    err,
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                error_handler.collect_error(
                    SchemaErrorReason.CHECK_ERROR,
                    SchemaError(
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

