"""Handle schema errors."""

import ibis

from pandera.api.base.error_handler import ErrorHandler as _ErrorHandler


class ErrorHandler(_ErrorHandler):
    """Handler for schema- and data-level errors during validation."""

    @staticmethod
    def _count_failure_cases(failure_cases: ibis.Table) -> int:
        # Failure cases can be a table object or a scalar value. Try
        # getting the number of elements in failure cases or set to one.
        if isinstance(failure_cases, ibis.Table):
            return failure_cases.count().to_pyarrow().as_py()
        else:
            return 1
