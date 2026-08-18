"""Shared, dataframe-library-agnostic statistics helpers.

This module must stay free of pandas (and other dataframe-library) imports so
that pandas-free entrypoints like ``pandera.polars`` can use it.
"""

from __future__ import annotations

import warnings
from typing import Any, Union

from pandera.api.checks import Check


def string_length_check_statistics(
    min_len: int, max_len: int
) -> dict[str, Any]:
    """Build ``parse_check_statistics``-compatible stats for
    :meth:`~pandera.api.checks.Check.str_length`.
    """
    return {
        "str_length": {
            "min_value": min_len,
            "max_value": max_len,
        },
    }


def parse_check_statistics(check_stats: Union[dict[str, Any], None]):
    """Convert check statistics to a list of Check objects, including their options."""
    if check_stats is None:
        return None
    checks = []
    for check_name, stats in check_stats.items():
        check = getattr(Check, check_name)
        try:
            # Extract options if present
            if isinstance(stats, dict):
                options = (
                    stats.pop("options", {}) if "options" in stats else {}
                )
                if stats:  # If there are remaining stats
                    check_instance = check(**stats)
                else:  # Handle case where all stats were in options
                    check_instance = check()
                # Apply options to the check instance
                for option_name, option_value in options.items():
                    setattr(check_instance, option_name, option_value)
                checks.append(check_instance)
            else:
                # Handle unary check case
                checks.append(check(stats))
        except TypeError:
            # if stats cannot be unpacked as key-word args, assume unary check.
            checks.append(check(stats))
    return checks if checks else None


def parse_checks(checks) -> Union[list[dict[str, Any]], None]:
    """Convert Check object to check statistics including options."""

    def _has_custom_error(check: Check) -> bool:
        """Determine whether a check has a user-defined error message."""
        if check.error is None:
            return False

        if check.name is None or not Check.is_builtin_check(check.name):
            return True

        try:
            default_check = getattr(Check, check.name)(
                **(check.statistics or {})
            )
        except (AttributeError, TypeError, ValueError):
            return True

        return check.error != default_check.error

    check_statistics = []

    for check in checks:
        if check not in Check:
            warnings.warn(
                "Only registered checks may be serialized to statistics. "
                "Did you forget to register it with the extension API? "
                f"Check `{check.name}` will be skipped."
            )
            continue

        # Get base statistics
        base_stats = {} if check.statistics is None else check.statistics

        # Collect check options
        check_options = {
            "check_name": check.name,
            "raise_warning": check.raise_warning,
            "n_failure_cases": check.n_failure_cases,
            "ignore_na": check.ignore_na,
        }
        if _has_custom_error(check):
            check_options["error"] = check.error

        # Filter out None values from options
        check_options = {
            k: v for k, v in check_options.items() if v is not None
        }

        # Combine statistics with options
        if check_options:
            base_stats["options"] = check_options
            check_statistics.append(base_stats)

    return check_statistics if check_statistics else None
