"""Backend implementation for PySpark schema components."""

import re
from collections.abc import Iterable
from copy import copy

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pandera.api.base.error_handler import ErrorCategory, ErrorHandler
from pandera.backends.pyspark.column import ColumnSchemaBackend
from pandera.backends.pyspark.error_formatters import scalar_failure_case
from pandera.errors import SchemaError
from pandera.validation_depth import ValidationScope, validate_scope


class ColumnBackend(ColumnSchemaBackend):
    """Backend implementation for PySpark dataframe columns."""

    def validate(
        self,
        check_obj: DataFrame,
        schema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> DataFrame:
        """Validation backend implementation for PySpark dataframe columns."""

        error_handler = ErrorHandler(lazy=lazy)

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

            except SchemaError as err:
                error_handler.collect_error(
                    ErrorCategory.DATA, err.reason_code, err
                )

        column_keys_to_check = (
            self.get_regex_columns(schema, check_obj)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            if schema.coerce:
                check_obj = self.coerce_dtype(check_obj, schema=schema)
            validate_column(check_obj, column_name)

        return check_obj

    def get_regex_columns(self, schema, check_obj) -> Iterable:
        """Get matching column names based on regex column name pattern.

        :param schema: schema specification to use
        :param columns: columns to regex pattern match
        :returns: matching columns
        """
        columns = check_obj.columns
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

        return column_keys_to_check

    @validate_scope(scope=ValidationScope.SCHEMA)
    def coerce_dtype(
        self,
        check_obj: DataFrame,
        *,
        schema,
    ) -> DataFrame:
        """Coerce dtype of a column, handling duplicate column names."""

        check_obj = check_obj.withColumn(
            schema.name, col(schema.name).cast(schema.dtype)
        )

        return check_obj
