import traceback

from pandera.backends.pandas.container import PandasSchemaContainerBackend
from pandera.backends.pandas.field import PandasSchemaFieldBackend
from pandera.errors import SchemaError, SchemaErrors
from pandera.error_formatters import scalar_failure_case


class PandasSchemaFieldComponentBackend(PandasSchemaFieldBackend):
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
            raise SchemaErrors(
                error_handler.collected_errors, check_obj
            )
        return check_results


class PandasSchemaIndexComponentBackend(PandasSchemaFieldBackend):
    pass


class PandasSchemaMultiIndexComponentBackend(PandasSchemaContainerBackend):
    pass
