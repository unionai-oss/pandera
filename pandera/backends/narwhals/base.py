"""Base schema backend for narwhals."""

import warnings

import narwhals.stable.v1 as nw

from pandera.api.narwhals.utils import _to_native
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import SchemaErrorReason, SchemaWarning


def _materialize(frame) -> nw.DataFrame:
    """Materialize a LazyFrame or SQL-lazy DataFrame to a narwhals DataFrame.

    Delegates to NarwhalsCheckBackend._materialize — single implementation,
    no duplication. _materialize stays in checks.py per locked design decision.
    """
    return NarwhalsCheckBackend._materialize(frame)


class NarwhalsSchemaBackend(BaseSchemaBackend):
    """Base schema backend for narwhals-backed DataFrames.

    Provides shared helpers used by ColumnBackend (components.py) and
    future container-level backends (Phase 4).
    """

    def subsample(
        self,
        check_obj,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
    ):
        """Return a (possibly subsampled) version of check_obj.

        :param head: Number of rows to take from the head.
        :param tail: Number of rows to take from the tail.
        :param sample: Not supported — raises NotImplementedError.
        :param random_state: Ignored (no random sampling supported).
        :raises NotImplementedError: If sample is not None.
        """
        if sample is not None:
            raise NotImplementedError(
                "sample= is not supported in the narwhals backend. "
                "Use head= or tail= instead."
            )

        obj_subsample = []
        if head is not None:
            obj_subsample.append(_materialize(check_obj).head(head))
        if tail is not None:
            obj_subsample.append(_materialize(check_obj).tail(tail))

        if not obj_subsample:
            return check_obj

        result = nw.concat(obj_subsample).unique()
        return result

    def run_check(self, check_obj, schema, check, check_index, *args):
        """Execute a single Check object and return a CoreCheckResult.

        :param check_obj: The narwhals LazyFrame or DataFrame to check.
        :param schema: The schema object owning this check.
        :param check: The Check instance to run.
        :param check_index: Index of the check within schema.checks.
        :param args: Extra arguments forwarded to the check callable.
        :returns: CoreCheckResult with passed, failure_cases, etc.
        """
        check_result = check(check_obj, *args)

        # Materialize the passed frame
        passed_df = _materialize(check_result.check_passed)
        passed = bool(passed_df[CHECK_OUTPUT_KEY][0])

        message = None
        failure_cases = None

        if not passed:
            if check_result.failure_cases is None:
                failure_cases = passed
                message = (
                    f"Check '{check}' failed — no failure cases captured."
                )
            else:
                fc = _materialize(check_result.failure_cases)
                # Drop CHECK_OUTPUT_KEY column if present
                if CHECK_OUTPUT_KEY in fc.collect_schema().names():
                    fc = fc.drop(CHECK_OUTPUT_KEY)
                failure_cases = _to_native(fc)
                message = (
                    f"Check '{check}' failed. "
                    f"Failure cases: {fc.head().rows(named=True)}"
                )

            if check.raise_warning:
                warnings.warn(message, SchemaWarning)
                return CoreCheckResult(
                    passed=True,
                    check=check,
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                )

        check_output_df = _materialize(check_result.check_output)
        return CoreCheckResult(
            passed=passed,
            check=check,
            check_index=check_index,
            check_output=_to_native(check_output_df),
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            message=message,
            failure_cases=failure_cases,
        )

    def is_float_dtype(self, check_obj, col_name: str) -> bool:
        """Return True if the column col_name has a float dtype.

        Uses collect_schema() so it works on both LazyFrame and DataFrame
        without triggering full materialization.

        :param check_obj: narwhals LazyFrame or DataFrame.
        :param col_name: Name of the column to inspect.
        :returns: True if the column dtype is a floating-point type.
        """
        return check_obj.collect_schema()[col_name].is_float()
