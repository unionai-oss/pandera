"""Check backend for TensorDict."""

from functools import partial
from typing import Any

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.backends.base import BaseCheckBackend, CoreCheckResult
from pandera.errors import SchemaErrorReason

_MAX_FAILURE_CASES = 10


def _failing_element_indices(mask: Any) -> list[Any]:
    """Compute the indices of failing elements in a boolean mask tensor.

    Returns a flat list of indices. For 1-D masks this is a list of integer
    indices; for N-D masks this is a list of index tuples.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return []
    if not isinstance(mask, torch.Tensor):
        return []
    if mask.ndim == 0:
        # A scalar mask (whole-tensor reduction) has no per-element index.
        return []
    if mask.dtype != torch.bool:
        mask = mask.bool()
    nonzero = torch.nonzero(mask)
    if nonzero.numel() == 0:
        return []
    indices = nonzero.tolist()
    if mask.ndim == 1:
        return [i[0] for i in indices]
    return [tuple(i) for i in indices]


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

    def preprocess(self, check_obj: Any, key: Any = None) -> Any:
        """Preprocesses the check object before applying the check function."""
        # For tensor data, no preprocessing needed
        return check_obj

    def _extract_failure_cases(
        self, check_obj: Any, result: Any
    ) -> dict[str, Any] | None:
        """Extract failing elements and their indices from a check result.

        ``result`` is the raw output of the check function. When it is a
        boolean tensor with the same shape as the checked tensor, each element
        maps to a per-element failure case (mirroring dataframe leaf-case
        reports). Otherwise (scalar booleans, or non-mask outputs) no per-
        element cases are available.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            return None
        if not isinstance(check_obj, torch.Tensor):
            return None
        if not isinstance(result, torch.Tensor):
            return None
        if result.dtype != torch.bool:
            return None
        if result.ndim == 0:
            # A scalar result means the check reduced the whole tensor, so
            # there is no per-element location to report.
            return None

        leading_mask = bool(
            result.ndim < check_obj.ndim
            and result.shape == check_obj.shape[: result.ndim]
        )
        elementwise_mask = result.shape == check_obj.shape
        if not (leading_mask or elementwise_mask):
            # Output is not a mask aligned with the checked tensor (e.g. a
            # scalar boolean from a check that reduces the whole tensor), so
            # no per-element failure cases are available.
            return None

        mask = ~result
        try:
            n_failures = int(mask.sum().item())
        except Exception:
            return None
        if n_failures == 0:
            return None

        indices = _failing_element_indices(mask)
        # Boolean indexing flattens to the failing elements (or leading
        # elements, for a reduced leading-dimension mask) in the same order as
        # ``torch.nonzero``.
        failure_values = check_obj[mask]
        try:
            values = [v.item() for v in failure_values]
        except Exception:
            values = [str(v) for v in failure_values]

        # Respect the failure-case display limit.
        shown = min(n_failures, _MAX_FAILURE_CASES)
        return {
            "index": indices[:shown],
            "failure_case": values[:shown],
            "n_failure_cases": n_failures,
        }

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

        # Apply the check function. Exceptions propagate to the caller
        # (``TensorDictSchemaBackend._run_value_checks``), which records the
        # underlying message in the resulting :class:`SchemaError`.
        result = self.check_fn(check_obj)

        # Reduce to boolean
        passed = self._reduce_to_bool(result)

        if not passed:
            failure_cases = self._extract_failure_cases(check_obj, result)
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

    def postprocess(
        self, check_obj: Any, check_output: CheckResult
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        if isinstance(check_output, CheckResult):
            return check_output
        if isinstance(check_output, CoreCheckResult):
            return CheckResult(
                check_output=check_output.check_output,
                check_passed=check_output.check_passed,
                checked_object=check_obj,
                failure_cases=check_output.failure_cases,
            )
        return CheckResult(
            check_output=check_output,
            check_passed=bool(check_output),
            checked_object=check_obj,
            failure_cases=None,
        )

    def failure_cases_message(
        self, failure_cases: dict[str, Any] | None
    ) -> str:
        """Format a human-readable failure-cases message.

        :param failure_cases: structured failure cases from
            ``_extract_failure_cases``.
        :returns: message string describing where the failure occurred.
        """
        if failure_cases is None:
            return "check failed"
        indices = failure_cases.get("index", []) or []
        n = failure_cases.get(
            "n_failure_cases", len(failure_cases.get("failure_case", []))
        )
        if indices:
            shown_indices = ", ".join(str(i) for i in indices[:5])
            suffix = f" (+{n - len(indices[:5])} more)" if n > 5 else ""
            return (
                f"failure cases found at tensor index "
                f"[{shown_indices}]{suffix}"
            )
        values = failure_cases.get("failure_case", []) or []
        shown = ", ".join(str(v) for v in values[:5])
        return f"failure cases: {shown}"
