"""TensorDict parsing, validation, and error-reporting backends."""

import copy
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
        from tensordict import TensorDict, tensorclass

        return isinstance(obj, (TensorDict, tensorclass))
    except ImportError:
        return False


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
        lazy: bool = False,
        inplace: bool = False,
    ):
        """Validate a TensorDict against the schema."""
        error_handler = ErrorHandler(lazy)

        if not _is_tensordict(check_obj):
            raise TypeError(
                f"Expected TensorDict or tensorclass, got {type(check_obj)}"
            )

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
        self._check_dtypes_and_shapes(
            check_obj, schema, error_handler, lazy
        )
        self._run_value_checks(check_obj, schema, error_handler, lazy)

        return error_handler

    def _check_batch_size(
        self, check_obj, schema, error_handler, lazy
    ):
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
                schema_context={"batch_size": actual_batch_size},
                message=error_msg,
                reason_code=SchemaErrorReason.WRONG_TYPE,
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
                error_msg = (
                    f"Expected batch_size[{i}]={exp}, got batch_size[{i}]={act}"
                )
                error = SchemaError(
                    schema=schema,
                    schema_context={"batch_size": actual_batch_size},
                    message=error_msg,
                    reason_code=SchemaErrorReason.WRONG_TYPE,
                )
                error_handler.collect_error(
                    get_error_category(error.reason_code),
                    error.reason_code,
                    error,
                )

    def _check_keys(self, check_obj, schema, error_handler, lazy):
        """Validate required keys exist."""
        for key in schema.columns:
            if key not in check_obj:
                error_msg = f"Missing key '{key}' in TensorDict"
                error = SchemaError(
                    schema=schema,
                    schema_context={"keys": list(check_obj.keys())},
                    message=error_msg,
                    reason_code=SchemaErrorReason.MISSING_COLUMN,
                )
                error_handler.collect_error(
                    get_error_category(error.reason_code),
                    error.reason_code,
                    error,
                )

    def _check_dtypes_and_shapes(
        self, check_obj, schema, error_handler, lazy
    ):
        """Validate tensor dtypes and shapes."""
        for key, tensor_schema in schema.columns.items():
            if key not in check_obj:
                continue

            tensor = check_obj[key]

            if not isinstance(tensor, torch.Tensor):
                error_msg = f"Key '{key}' is not a torch.Tensor"
                error = SchemaError(
                    schema=schema,
                    schema_context={"key": key, "type": type(tensor)},
                    message=error_msg,
                    reason_code=SchemaErrorReason.WRONG_TYPE,
                )
                error_handler.collect_error(
                    get_error_category(error.reason_code),
                    error.reason_code,
                    error,
                )
                continue

            if tensor_schema.dtype is not None:
                expected_dtype = tensor_schema.dtype.type
                if tensor.dtype != expected_dtype:
                    error_msg = (
                        f"Key '{key}': expected dtype {expected_dtype}, "
                        f"got {tensor.dtype}"
                    )
                    error = SchemaError(
                        schema=schema,
                        schema_context={
                            "key": key,
                            "dtype": str(tensor.dtype),
                        },
                        message=error_msg,
                        reason_code=SchemaErrorReason.WRONG_TYPE,
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
                        schema_context={
                            "key": key,
                            "shape": actual_shape,
                        },
                        message=error_msg,
                        reason_code=SchemaErrorReason.WRONG_TYPE,
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
                            schema_context={
                                "key": key,
                                "shape": actual_shape,
                            },
                            message=error_msg,
                            reason_code=SchemaErrorReason.WRONG_TYPE,
                        )
                        error_handler.collect_error(
                            get_error_category(error.reason_code),
                            error.reason_code,
                            error,
                        )

    def _run_value_checks(
        self, check_obj, schema, error_handler, lazy
    ):
        """Run value checks on tensor values."""
        from pandera.api.base.checks import CheckResult

        for key, tensor_schema in schema.columns.items():
            if key not in check_obj:
                continue

            if not tensor_schema.checks:
                continue

            tensor = check_obj[key]
            is_tensorclass = hasattr(tensor, "as_dict")
            tensor_data = (
                tensor.as_dict() if is_tensorclass else {key: tensor}
            )

            for check_index, check in enumerate(tensor_schema.checks):
                try:
                    check_result = check.get_backend(check_obj)(check)(
                        tensor_data, key
                    )
                except Exception as exc:
                    check_result = CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_FAILED,
                        message=str(exc),
                    )

                if not check_result.passed:
                    error_msg = (
                        f"Check '{check}' failed for key '{key}': "
                        f"{check_result.message}"
                    )
                    error = SchemaError(
                        schema=schema,
                        schema_context={"key": key},
                        message=error_msg,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_FAILED,
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
        raise NotImplementedError(
            "Use _run_value_checks instead"
        )

    def run_schema_component_checks(
        self, check_obj, schema, schema_components, lazy
    ):
        """Run checks for all schema components."""
        raise NotImplementedError(
            "Use _run_value_checks instead"
        )

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
        return None

    def drop_invalid_rows(self, check_obj, error_handler):
        """Remove invalid elements in a check_obj."""
        raise NotImplementedError(
            "drop_invalid_rows is not applicable for TensorDict"
        )
