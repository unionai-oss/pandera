"""TensorDict parsing, validation, and error-reporting backends."""

import copy
from collections import defaultdict
from typing import Any

from pandera import errors
from pandera.api.base.error_handler import ErrorHandler, get_error_category
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.errors import SchemaError, SchemaErrorReason

try:
    import torch
except ImportError:
    torch = None


def _is_tensordict(obj: Any) -> bool:
    """Check if object is a TensorDict or tensorclass."""
    if torch is None:
        return False
    try:
        from tensordict import TensorDict
        
        # Check for TensorDict first (more common case)
        if isinstance(obj, TensorDict):
            return True
        
        # Check for tensorclass - use _is_tensorclass attribute which is set
        # on all tensorclass instances
        if hasattr(obj, '_is_tensorclass') and obj._is_tensorclass:
            return True
            
        return False
    except ImportError:
        return False


def _get_tensor(obj: Any, key: str) -> Any:
    """Get tensor from TensorDict or tensorclass object."""
    # For TensorDict, use __getitem__
    try:
        return obj[key]
    except (KeyError, TypeError, ValueError):
        pass
    
    # For tensorclass, use attribute access
    if hasattr(obj, key):
        return getattr(obj, key)
    
    raise KeyError(f"Key '{key}' not found in object")


class TensorDictSchemaBackend(BaseSchemaBackend):
    """Backend for TensorDict schema validation."""

    def preprocess(self, check_obj, inplace: bool = False):
        """Preprocesses a check object before applying check functions."""
        if not inplace:
            if _is_tensordict(check_obj):
                check_obj = check_obj.clone()
            else:
                check_obj = check_obj.copy()
        return check_obj

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        """Validate a TensorDict against the schema."""
        error_handler = ErrorHandler(lazy)

        check_obj = self.preprocess(check_obj, inplace=inplace)

        error_handler = self.run_checks_and_handle_errors(
            error_handler, schema, check_obj, lazy
        )

        if error_handler.collected_errors:
            raise errors.SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    def coerce_dtype(self, check_obj, schema=None):
        """Coerce the dtype of tensors in the TensorDict."""
        return check_obj

    def run_checks_and_handle_errors(
        self, error_handler, schema, check_obj, lazy
    ):
        """Run all checks and collect errors."""
        self._check_batch_size(check_obj, schema, error_handler, lazy)
        self._check_keys(check_obj, schema, error_handler, lazy)
        self._check_dtypes_and_shapes(check_obj, schema, error_handler, lazy)
        self._run_value_checks(check_obj, schema, error_handler, lazy)

        return error_handler

    def _check_batch_size(self, check_obj, schema, error_handler, lazy):
        """Validate batch_size matches."""
        if schema.batch_size is None:
            return

        actual_batch_size = check_obj.batch_size
        expected_batch_size = schema.batch_size

        if len(actual_batch_size) != len(expected_batch_size):
            error_msg = (
                f"Expected batch_size {expected_batch_size}, "
                f"got {actual_batch_size}"
            )
            error = SchemaError(
                schema=schema,
                data=check_obj,
                message=error_msg,
                reason_code=SchemaErrorReason.WRONG_DATATYPE,
            )
            error_handler.collect_error(
                get_error_category(error.reason_code),
                error.reason_code,
                error,
            )
            return

        for i, (exp, act) in enumerate(
            zip(expected_batch_size, actual_batch_size)
        ):
            if exp is not None and exp != act:
                error_msg = f"Expected batch_size[{i}]={exp}, got batch_size[{i}]={act}"
                error = SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=error_msg,
                    reason_code=SchemaErrorReason.WRONG_DATATYPE,
                )
                error_handler.collect_error(
                    get_error_category(error.reason_code),
                    error.reason_code,
                    error,
                )

    def _check_keys(self, check_obj, schema, error_handler, lazy):
        """Validate required keys exist."""
        for key in schema.keys:
            # Check if key exists - use different methods for TensorDict vs tensorclass
            is_key_present = False
            
            try:
                from tensordict import TensorDict
                
                # For TensorDict, use 'in' operator
                if isinstance(check_obj, TensorDict):
                    is_key_present = key in check_obj
                else:
                    # For tensorclass, use keys() method or attribute access
                    if hasattr(check_obj, 'keys'):
                        is_key_present = key in check_obj.keys()
                    elif hasattr(check_obj, key):
                        is_key_present = True
            except ImportError:
                pass
            
            if not is_key_present:
                error_msg = f"Missing key '{key}' in TensorDict"
                error = SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=error_msg,
                    reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                )
                error_handler.collect_error(
                    get_error_category(error.reason_code),
                    error.reason_code,
                    error,
                )

    def _check_dtypes_and_shapes(self, check_obj, schema, error_handler, lazy):
        """Validate tensor dtypes and shapes."""
        for key, tensor_schema in schema.keys.items():
            try:
                tensor = _get_tensor(check_obj, key)
            except KeyError:
                # Key doesn't exist (should already be caught by _check_keys)
                continue

            if not isinstance(tensor, torch.Tensor):
                error_msg = f"Key '{key}' is not a torch.Tensor"
                error = SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=error_msg,
                    reason_code=SchemaErrorReason.WRONG_DATATYPE,
                )
                error_handler.collect_error(
                    get_error_category(error.reason_code),
                    error.reason_code,
                    error,
                )
                continue

            if tensor_schema.dtype is not None:
                expected_dtype = tensor_schema.dtype.type
                actual_dtype = tensor.dtype
                if actual_dtype != expected_dtype:
                    error_msg = (
                        f"Key '{key}': expected dtype {expected_dtype}, "
                        f"got {actual_dtype}"
                    )
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=error_msg,
                        reason_code=SchemaErrorReason.WRONG_DATATYPE,
                    )
                    error_handler.collect_error(
                        get_error_category(error.reason_code),
                        error.reason_code,
                        error,
                    )

            if tensor_schema.shape is not None:
                actual_shape = tuple(tensor.shape)
                expected_shape = tensor_schema.shape

                if len(actual_shape) != len(expected_shape):
                    error_msg = (
                        f"Key '{key}': expected shape {expected_shape}, "
                        f"got {actual_shape}"
                    )
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=error_msg,
                        reason_code=SchemaErrorReason.WRONG_DATATYPE,
                    )
                    error_handler.collect_error(
                        get_error_category(error.reason_code),
                        error.reason_code,
                        error,
                    )
                    continue

                for i, (exp, act) in enumerate(
                    zip(expected_shape, actual_shape)
                ):
                    if exp is not None and exp != act:
                        error_msg = (
                            f"Key '{key}': expected shape[{i}]={exp}, "
                            f"got shape[{i}]={act}"
                        )
                        error = SchemaError(
                            schema=schema,
                            data=check_obj,
                            message=error_msg,
                            reason_code=SchemaErrorReason.WRONG_DATATYPE,
                        )
                        error_handler.collect_error(
                            get_error_category(error.reason_code),
                            error.reason_code,
                            error,
                        )

    def _run_value_checks(self, check_obj, schema, error_handler, lazy):
        """Run value checks on tensor values."""
        from pandera.api.base.checks import CheckResult

        for key, tensor_schema in schema.keys.items():
            try:
                tensor = _get_tensor(check_obj, key)
            except KeyError:
                continue

            if not tensor_schema.checks:
                continue

            # For value checks, we need to pass the tensor data
            tensor_data = {key: tensor}

            for check_index, check in enumerate(tensor_schema.checks):
                try:
                    # Use TensorDictCheckBackend directly
                    from pandera.backends.tensordict.checks import TensorDictCheckBackend
                    
                    check_backend = TensorDictCheckBackend(check)
                    
                    # Apply check to the tensor (not dict)
                    check_result = check_backend.apply(tensor)
                    
                    # Postprocess result
                    check_result = check_backend.postprocess(tensor, check_result)
                except Exception as exc:
                    check_result = CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=str(exc),
                    )

                if not check_result.check_passed:
                    error_msg = (
                        f"Check '{check}' failed for key '{key}': check failed"
                    )
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=error_msg,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                    )
                    error_handler.collect_error(
                        get_error_category(error.reason_code),
                        error.reason_code,
                        error,
                    )

    def run_check(self, check_obj, schema, check, check_index, *args):
        """Run a single check on the check object."""
        raise NotImplementedError("Use _run_value_checks instead")

    def run_checks(self, check_obj, schema):
        """Run a list of checks on the check object."""
        raise NotImplementedError("Use _run_value_checks instead")

    def run_schema_component_checks(
        self, check_obj, schema, schema_components, lazy
    ):
        """Run checks for all schema components."""
        raise NotImplementedError("Use _run_value_checks instead")

    def check_name(self, check_obj, schema):
        """Core check that checks the name of the check object."""
        raise NotImplementedError

    def check_nullable(self, check_obj, schema):
        """Core check that checks the nullability of a check object."""
        raise NotImplementedError

    def check_unique(self, check_obj, schema):
        """Core check that checks the uniqueness of values in a check object."""
        raise NotImplementedError

    def check_dtype(self, check_obj, schema):
        """Core check that checks the data type of a check object."""
        raise NotImplementedError

    def failure_cases_metadata(
        self, schema_name: str, schema_errors: list[SchemaError]
    ):
        """Get failure cases metadata for lazy validation."""
        from collections import defaultdict

        from pandera.api.base.error_handler import ErrorHandler
        from pandera.errors import FailureCaseMetadata

        error_dicts = {}
        error_handler = ErrorHandler()
        error_handler.collect_errors(schema_errors)

        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema_name)
            error_dicts = dict(error_dicts)

        failure_cases = [
            err.failure_cases for err in schema_errors if err.failure_cases
        ]

        error_counts: dict[str, int] = defaultdict(int)
        for error in error_handler.collected_errors:
            error_counts[error["reason_code"].name] += 1

        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=error_dicts,
            error_counts=dict(error_counts),
        )

    def drop_invalid_rows(self, check_obj, error_handler):
        """Remove invalid elements in a check_obj."""
        raise NotImplementedError(
            "drop_invalid_rows is not applicable for TensorDict"
        )
