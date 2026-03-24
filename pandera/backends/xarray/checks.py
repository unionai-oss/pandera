"""Apply :class:`~pandera.api.checks.Check` instances to xarray objects.

Builtin checks return a boolean :class:`~xarray.DataArray` mask or a scalar
``bool`` (for dataset-wide aggregation). ``ignore_na`` ORs null positions into
that mask so they do not fail vectorized checks.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.backends.base import BaseCheckBackend


class XarrayCheckBackend(BaseCheckBackend):
    """Evaluate :class:`~pandera.api.checks.Check` on xarray objects."""

    def __init__(self, check: Check):
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def preprocess(self, check_obj, key: str | None):
        import xarray as xr

        if key is None:
            return check_obj
        if isinstance(check_obj, xr.Dataset):
            return check_obj[key]
        raise TypeError(
            f"cannot select key {key!r} on {type(check_obj).__name__}"
        )

    def apply(self, check_obj):
        if self.check.element_wise:
            return self._apply_elementwise(check_obj)
        return self.check_fn(check_obj)

    def _apply_elementwise(self, check_obj):
        import xarray as xr

        if not isinstance(check_obj, xr.DataArray):
            raise NotImplementedError(
                "element_wise checks on non-DataArray xarray objects."
            )

        def _eval_scalar(x):
            if self.check.ignore_na:
                try:
                    if isinstance(x, (float, np.floating)) and np.isnan(x):
                        return True
                except TypeError:
                    pass
            return bool(self.check_fn(x))

        vectorized = np.vectorize(_eval_scalar, otypes=[bool])
        out = vectorized(check_obj.values)
        return xr.DataArray(out, dims=check_obj.dims, coords=check_obj.coords)

    def postprocess(self, check_obj, check_output) -> CheckResult:
        import xarray as xr

        if isinstance(check_output, (bool, np.bool_)):
            return CheckResult(
                check_output=check_output,
                check_passed=bool(check_output),
                checked_object=check_obj,
                failure_cases=None,
            )
        if isinstance(check_output, xr.DataArray):
            if check_output.dtype == np.dtype(bool):
                mask = check_output
            else:
                mask = check_output.astype(bool)
            # Treat missing values as passing when ignore_na=True (pandas parity).
            if self.check.ignore_na and isinstance(check_obj, xr.DataArray):
                mask = mask | check_obj.isnull()
            passed = bool(np.all(mask.values))
            failures = None
            if not passed and isinstance(check_obj, xr.DataArray):
                failures = check_obj.where(~mask)
            return CheckResult(
                check_output=mask,
                check_passed=passed,
                checked_object=check_obj,
                failure_cases=failures,
            )
        raise TypeError(
            f"check_fn returned unsupported type: {type(check_output)!r}"
        )

    def __call__(self, check_obj, key: str | None = None) -> CheckResult:
        prepped = self.preprocess(check_obj, key)
        out = self.apply(prepped)
        return self.postprocess(prepped, out)
