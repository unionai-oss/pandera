"""Make schema error messages human-friendly."""

from typing import Any, Optional

from pandera.backends.error_formatters import format_failure_cases_with_truncation
from pandera.config import get_config_context


def format_generic_error_message(
    parent_schema,
    check,
) -> str:
    """Construct an error message when a check validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    """
    return f"{parent_schema} failed validation " f"{check.error}"


def scalar_failure_case(x) -> dict:
    """Construct failure case from a scalar value.

    :param x: a scalar value representing failure case.
    :returns: Dictionary used for error reporting with ``SchemaErrors``.
    """
    return {
        "index": [None],
        "failure_case": [x],
    }


def format_failure_cases_message(
    failure_cases: Any,
    max_reported_failures: Optional[int] = None,
) -> str:
    """Format failure cases for PySpark error messages.
    
    Note: PySpark currently only supports scalar failure cases.
    This function is provided for consistency with other backends
    and future extensibility.
    
    :param failure_cases: PySpark DataFrame or dict containing failure cases
    :param max_reported_failures: Maximum number of failures to report.
        If None, uses config value.
    :return: Formatted failure cases string
    """
    if max_reported_failures is None:
        config = get_config_context()
        max_reported_failures = config.max_reported_failures
    
    # PySpark currently only handles scalar failures
    # This is a placeholder for future vectorized failure case support
    if isinstance(failure_cases, dict):
        if "failure_case" in failure_cases:
            return str(failure_cases["failure_case"])
    
    return str(failure_cases)
