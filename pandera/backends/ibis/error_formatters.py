"""Make schema error messages human-friendly for Ibis."""

from typing import Any, Optional

import pandas as pd

from pandera.backends.error_formatters import (
    format_failure_cases_with_truncation,
)
from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
    reshape_failure_cases,
)
from pandera.config import get_config_context


def format_vectorized_error_message(
    parent_schema,
    check,
    check_index: int,
    reshaped_failure_cases: Any,
    max_reported_failures: Optional[int] = None,
) -> str:
    """Construct an error message when a validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    :param reshaped_failure_cases: The failure cases encountered by the
        element-wise or vectorized validator.
    :param max_reported_failures: Maximum number of failures to report
        in the error message. If None, use config value.
    """
    if max_reported_failures is None:
        config = get_config_context()
        max_reported_failures = config.max_reported_failures

    import re

    pattern = r"<Check\s+([^:>]+):\s*([^>]+)>"
    matches = re.findall(pattern, str(check))

    check_strs = [f"{match[1]}" for match in matches]

    if check_strs:
        check_str = check_strs[0]
    else:
        check_str = str(check)

    failure_cases = reshaped_failure_cases.failure_case
    total_failures = len(failure_cases)

    def format_all(cases):
        return ", ".join(cases.apply(str))

    def format_limited(cases, limit):
        limited = cases.iloc[:limit]
        formatted = ", ".join(limited.apply(str))
        return formatted, len(limited)

    failure_cases_string = format_failure_cases_with_truncation(
        failure_cases,
        total_failures,
        max_reported_failures,
        format_all,
        format_limited,
    )

    return (
        f"{parent_schema.__class__.__name__} '{parent_schema.name}' failed "
        f"element-wise validator number {check_index}: "
        f"{check_str} failure cases: {failure_cases_string}"
    )


# Re-export functions that don't need modification
__all__ = [
    "format_generic_error_message",
    "format_vectorized_error_message",
    "reshape_failure_cases",
]
