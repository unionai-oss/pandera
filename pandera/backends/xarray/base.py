"""Base backend helpers for xarray validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pandera.api.base.error_handler import ErrorHandler
from pandera.backends.base import (
    BaseSchemaBackend,
    CoreCheckResult,
    CoreParserResult,
)
from pandera.backends.pandas.error_formatters import (
    format_generic_error_message,
)
from pandera.backends.xarray.error_formatters import (
    format_xarray_vectorized_error_message,
)
from pandera.config import ValidationScope
from pandera.errors import (
    FailureCaseMetadata,
    ParserError,
    SchemaError,
    SchemaErrorReason,
)
from pandera.validation_depth import validate_scope


class XarraySchemaBackend(BaseSchemaBackend):
    """Shared parser/check helpers for xarray schema backends."""

    def subsample(
        self,
        check_obj,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
    ):
        """Restrict data checks to a subset along the first dimension.

        Indices from ``head``, ``tail``, and ``sample`` are merged and
        de-duplicated. Scalar (0-D) arrays are returned unchanged. Only
        :class:`~xarray.DataArray` inputs are subsampled; Dataset validation
        passes the object through here.
        """
        import xarray as xr

        if not isinstance(check_obj, xr.DataArray):
            return check_obj
        if head is None and tail is None and sample is None:
            return check_obj
        if check_obj.ndim < 1:
            return check_obj
        dim = check_obj.dims[0]
        n = check_obj.sizes[dim]
        idxs: list[np.ndarray] = []
        if head is not None:
            idxs.append(np.arange(min(head, n)))
        if tail is not None:
            idxs.append(np.arange(max(0, n - tail), n))
        if sample is not None:
            rng = np.random.default_rng(random_state)
            k = min(sample, n)
            idxs.append(rng.choice(n, size=k, replace=False))
        combined = np.unique(np.sort(np.concatenate(idxs)))
        return check_obj.isel({dim: combined})

    def run_parser(
        self,
        check_obj,
        parser,
        parser_index: int,
    ) -> CoreParserResult:
        from pandera.api.parsers import Parser

        assert isinstance(parser, Parser)
        parser_result = parser(check_obj, None)
        return CoreParserResult(
            passed=True,
            parser=parser,
            parser_index=parser_index,
            parser_output=parser_result.parser_output,
            reason_code=SchemaErrorReason.DATAFRAME_PARSER,
            failure_cases=None,
            message=None,
        )

    def run_parsers(self, schema, check_obj: Any):
        for parser_index, parser in enumerate(schema.parsers):
            result = self.run_parser(check_obj, parser, parser_index)
            check_obj = result.parser_output
        return check_obj

    @validate_scope(scope=ValidationScope.DATA)
    def run_check(
        self,
        check_obj,
        schema,
        check,
        check_index: int,
        *args,
    ) -> CoreCheckResult:
        key = args[0] if args else None
        check_result = check(check_obj, key)
        passed = check_result.check_passed
        if not passed and check.raise_warning:
            return CoreCheckResult(
                passed=True,
                check=check,
                check_index=check_index,
                reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            )
        message = None
        failure_cases = check_result.failure_cases
        if not passed:
            if failure_cases is None:
                message = format_generic_error_message(
                    schema, check, check_index
                )
            else:
                import xarray as xr

                if isinstance(failure_cases, xr.DataArray):
                    message = format_xarray_vectorized_error_message(
                        schema, check, check_index, failure_cases
                    )
                else:
                    message = format_generic_error_message(
                        schema, check, check_index
                    )
        return CoreCheckResult(
            passed=passed,
            check=check,
            check_index=check_index,
            check_output=check_result.check_output,
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            message=message,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj, schema) -> list[CoreCheckResult]:
        results: list[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        None,
                    )
                )
            except Exception as err:
                err_msg = f'"{err.args[0]}"' if err.args else ""
                msg = f"{err.__class__.__name__}({err_msg})"
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=msg,
                        original_exc=err,
                    )
                )
        return results

    def coerce_dtype(self, check_obj: Any, schema=None):
        if schema is None or schema.dtype is None or not schema.coerce:
            return check_obj
        from pandera.engines import xarray_engine

        try:
            pandera_dtype = xarray_engine.Engine.dtype(schema.dtype)
            return pandera_dtype.try_coerce(check_obj)
        except ParserError as exc:
            raise SchemaError(
                schema,
                data=check_obj,
                message=f"Error coercing to {schema.dtype}: {exc}",
                failure_cases=str(exc.failure_cases),
                check=f"coerce_dtype('{schema.dtype}')",
                reason_code=SchemaErrorReason.DATATYPE_COERCION,
            ) from exc

    def failure_cases_metadata(
        self,
        schema_name: str,
        schema_errors: list[SchemaError],
    ) -> FailureCaseMetadata:
        handler = ErrorHandler(lazy=True)
        handler.collect_errors(schema_errors)

        def _dd(obj):
            if isinstance(obj, defaultdict):
                return {k: _dd(v) for k, v in obj.items()}
            return dict(obj) if hasattr(obj, "items") else obj

        summary = _dd(handler.summarize(schema_name or "schema"))
        counts: dict[str, int] = defaultdict(int)
        for entry in handler.collected_errors:
            rc = entry.get("reason_code")
            if rc is not None:
                counts[rc.name] += 1
        return FailureCaseMetadata(
            failure_cases=[
                e["error"].__str__() for e in handler.collected_errors
            ],
            message=summary,
            error_counts=dict(counts),
        )

    def drop_invalid_rows(self, check_obj, error_handler):
        raise NotImplementedError(
            "drop_invalid_rows is not supported for xarray containers."
        )
