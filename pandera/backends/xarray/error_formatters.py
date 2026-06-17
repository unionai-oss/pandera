"""Make xarray schema check error messages human-friendly."""

from __future__ import annotations

import re
from typing import Any

from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
)


def _check_display_str(check: Any) -> str:
    """Match :mod:`pandera.backends.pandas.error_formatters` check labeling."""
    pattern = r"<Check\s+([^:>]+):\s*([^>]+)>"
    matches = re.findall(pattern, str(check))
    if matches:
        return matches[0][1]
    return str(check)


def _unique_failure_values_flat(
    failure_da: Any,
    n_failure_cases: int | None,
) -> list[Any]:
    """Order-preserving unique non-missing values from a failure DataArray."""
    import numpy as np

    flat = failure_da.values.ravel()
    filtered: list[Any] = []
    for x in flat:
        if x is None:
            continue
        try:
            if np.isscalar(x) and np.isnan(x):
                continue
        except (TypeError, ValueError):
            pass
        filtered.append(x)
    unique_vals = list(dict.fromkeys(filtered))
    if n_failure_cases is not None:
        unique_vals = unique_vals[:n_failure_cases]
    return unique_vals


def format_xarray_vectorized_error_message(
    parent_schema: Any,
    check: Any,
    check_index: int,
    failure_da: Any,
) -> str:
    """Message for a failed vectorized check with DataArray failure locations."""
    import xarray as xr

    if not isinstance(failure_da, xr.DataArray):
        raise TypeError(
            "expected failure_cases to be an xarray.DataArray, found "
            f"{type(failure_da)}"
        )
    values = _unique_failure_values_flat(failure_da, check.n_failure_cases)
    if not values:
        return format_generic_error_message(parent_schema, check, check_index)
    failure_cases_string = ", ".join(str(v) for v in values)
    check_str = _check_display_str(check)
    return (
        f"{parent_schema.__class__.__name__} '{parent_schema.name}' failed "
        f"element-wise validator number {check_index}: "
        f"{check_str} failure cases: {failure_cases_string}"
    )
