"""Check backend for TensorDict."""

from functools import partial
from typing import Any

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.backends.base import BaseCheckBackend, CoreCheckResult
from pandera.errors import SchemaErrorReason


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

    def preprocess(self, check_obj: Any) -> Any:
        """Preprocesses the check object before applying the check function."""
        # For tensor data, no preprocessing needed
        return check_obj

    def apply(self, check_obj: Any) -> CheckResult:
        """Apply the check function to the check object.
        
        :param check_obj: Tensor data to validate.
        :returns: CheckResult with passed status and output.
        """
        check_obj = self.preprocess(check_obj)
        
        if self.check_fn is None:
            return CheckResult(
                check_output=True,
                check_passed=True,
                checked_object=check_obj,
                failure_cases=None,
            )

        # Apply the check function
        try:
            result = self.check_fn(check_obj)
        except Exception as exc:
            return CoreCheckResult(
                passed=False,
                check=self.check,
                reason_code=SchemaErrorReason.CHECK_ERROR,
                message=str(exc),
            )
        
        # Reduce to boolean
        passed = self._reduce_to_bool(result)
        
        if not passed:
            failure_cases = self._get_failure_cases(check_obj, result)
        else:
            failure_cases = None
        
        return CheckResult(
            check_output=result,
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def _reduce_to_bool(self, result: Any) -> bool:
        """Reduce a boolean tensor/array to a single boolean value."""
        try:
            import torch

            if isinstance(result, torch.Tensor):
                return result.all().item()
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(result, np.ndarray):
                return result.all().item()
        except ImportError:
            pass

        # Assume it's already a boolean
        return bool(result)

    def _get_failure_cases(
        self, check_obj: Any, result: Any
    ) -> Any | None:
        """Get the failure cases from the check result."""
        try:
            import torch

            if isinstance(check_obj, torch.Tensor):
                # Get indices where check failed
                if isinstance(result, torch.Tensor):
                    failure_mask = ~result
                    return check_obj[failure_mask]
        except ImportError:
            pass

        return None

    def postprocess(self, check_obj: Any, check_output: CheckResult) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        if isinstance(check_output, CheckResult):
            return check_output
        return CheckResult(
            check_output=check_output,
            check_passed=bool(check_output),
            checked_object=check_obj,
            failure_cases=None,
        )
