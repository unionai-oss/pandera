"""Common error formatting utilities for all backends."""

from typing import Callable, Optional, TypeVar, Union, Tuple

T = TypeVar('T')


def format_failure_cases_with_truncation(
    failure_cases: T,
    total_failures: int,
    max_reported_failures: int,
    format_all_cases: Callable[[T], str],
    format_limited_cases: Callable[[T, int], Tuple[str, int]],
) -> str:
    """
    Format failure cases with truncation based on max_reported_failures.
    
    This function provides a unified way to handle failure case formatting
    across different backends (pandas, polars, etc.) while allowing each
    backend to maintain its specific formatting requirements.
    
    :param failure_cases: The failure cases to format (backend-specific type)
    :param total_failures: Total number of failures
    :param max_reported_failures: Maximum failures to report 
        (-1 for unlimited, 0 for summary only)
    :param format_all_cases: Function to format all cases without truncation
    :param format_limited_cases: Function to format limited number of cases,
        returns tuple of (formatted_string, actual_count_shown)
    :return: Formatted failure cases string with truncation message if needed
    """
    if max_reported_failures == -1:
        return format_all_cases(failure_cases)
    
    if max_reported_failures == 0:
        return f"... {total_failures} failure cases"
    
    formatted_str, shown_count = format_limited_cases(failure_cases, max_reported_failures)
    
    if shown_count < total_failures:
        omitted_count = total_failures - shown_count
        return f"{formatted_str} ... and {omitted_count} more failure cases ({total_failures} total)"
    
    return formatted_str