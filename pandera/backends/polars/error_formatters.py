"""Make schema error messages human-friendly for Polars."""

from typing import Any, Optional

from pandera.backends.error_formatters import (
    format_failure_cases_with_truncation,
)
from pandera.config import get_config_context


def format_failure_cases_message(
    failure_cases: Any,
    max_reported_failures: Optional[int] = None,
) -> str:
    """Format failure cases for Polars error messages.

    :param failure_cases: Polars DataFrame containing failure cases
    :param max_reported_failures: Maximum number of failures to report.
        If None, uses config value.
    :return: Formatted failure cases string
    """
    if max_reported_failures is None:
        config = get_config_context()
        max_reported_failures = config.max_reported_failures

    total_failures = failure_cases.height

    def format_all(cases):
        return cases.rows(named=True)

    def format_limited(cases, limit):
        limited = cases.head(limit)
        return limited.rows(named=True), limited.height

    return format_failure_cases_with_truncation(
        failure_cases,
        total_failures,
        max_reported_failures,
        format_all,
        format_limited,
    )
