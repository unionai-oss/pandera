"""Check backend for TensorDict."""

from functools import partial
from typing import Any

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.backends.base import BaseCheckBackend, CoreCheckResult
from pandera.errors import SchemaErrorReason

try:
    import torch
except ImportError:
    torch = None


class TensorDictCheckBackend(BaseCheckBackend):
    """Check backend for TensorDict."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        self.check = check
        self.check_fn = (
            partial(check._check_fn, **check._check_kwargs)
            if check._check_fn is not None
            else None
        )

    def apply(self, check_obj: Any):
        """Apply the check function to the tensor data."""
        if self.check_fn is None:
            return CheckResult(
                check_output=True,
                check_passed=True,
                checked_object=check_obj,
                failure_cases=None,
            )

        if self.check.element_wise:
            return self._apply_element_wise(check_obj)
        else:
            return self._apply_full_tensor(check_obj)

    def _apply_element_wise(self, check_obj: Any) -> CheckResult:
        """Apply check element-wise to tensor."""
        result = self.check_fn(check_obj)
        if isinstance(result, torch.Tensor):
            passed = result.all().item()
        else:
            passed = bool(result)
        if not passed:
            return CheckResult(
                check_output=result,
                check_passed=False,
                checked_object=check_obj,
                failure_cases=None,
            )
        return CheckResult(
            check_output=result,
            check_passed=True,
            checked_object=check_obj,
            failure_cases=None,
        )

    def _apply_full_tensor(self, check_obj: Any) -> CheckResult:
        """Apply check to full tensor."""
        result = self.check_fn(check_obj)
        if isinstance(result, torch.Tensor):
            passed = result.all().item()
        else:
            passed = bool(result)
        if not passed:
            return CheckResult(
                check_output=result,
                check_passed=False,
                checked_object=check_obj,
                failure_cases=None,
            )
        return CheckResult(
            check_output=result,
            check_passed=True,
            checked_object=check_obj,
            failure_cases=None,
        )

    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        if isinstance(check_output, CheckResult):
            return check_output
        return CheckResult(
            check_output=check_output,
            check_passed=bool(check_output),
            checked_object=check_obj,
            failure_cases=None,
        )
