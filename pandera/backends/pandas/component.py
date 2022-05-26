import traceback
from copy import copy
from typing import Optional, Union

import pandas as pd

from pandera.backends.pandas.container import PandasSchemaContainerBackend
from pandera.backends.pandas.field import PandasSchemaFieldBackend
from pandera.core.pandas.types import is_field, is_index, is_table
from pandera.error_formatters import scalar_failure_case
from pandera.errors import SchemaError, SchemaErrors


class PandasSchemaFieldComponentBackend(PandasSchemaFieldBackend):
    def validate(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        if not inplace:
            check_obj = check_obj.copy()

        if schema.name is None:
            raise SchemaError(
                schema,
                check_obj,
                "column name is set to None. Pass the ``name` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.",
            )

        def validate_column(check_obj, column_name):
            super(PandasSchemaFieldComponentBackend, self).validate(
                check_obj,
                copy(schema).set_name(column_name),
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )

        column_keys_to_check = (
            schema.get_regex_columns(check_obj.columns)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            if schema.coerce:
                check_obj[column_name] = self.coerce_dtype(
                    check_obj[column_name]
                )
            if is_table(check_obj[column_name]):
                for i in range(check_obj[column_name].shape[1]):
                    validate_column(
                        check_obj[column_name].iloc[:, [i]], column_name
                    )
            else:
                validate_column(check_obj, column_name)

        return check_obj

    def coerce_dtype(self, obj: Union[pd.DataFrame, pd.Series, pd.Index]):
        """Coerce dtype of a column, handling duplicate column names."""
        # pylint: disable=super-with-arguments
        # TODO: use singledispatchmethod here
        if is_field(obj) or is_index(obj):
            return super(PandasSchemaFieldComponentBackend, self).coerce_dtype(
                obj
            )
        return obj.apply(
            lambda x: super(
                PandasSchemaFieldComponentBackend, self
            ).coerce_dtype(x),
            axis="columns",
        )

    def run_checks(self, check_obj, schema, error_handler, lazy):
        check_results = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(
                        check_obj, schema, check, check_index, schema.name
                    )
                )
            except SchemaError as err:
                error_handler.collect_error("dataframe_check", err)
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                error_handler.collect_error(
                    "check_error",
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

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(error_handler.collected_errors, check_obj)
        return check_results


class PandasSchemaIndexComponentBackend(PandasSchemaFieldBackend):
    pass


class PandasSchemaMultiIndexComponentBackend(PandasSchemaContainerBackend):
    pass
