"""Base schema backend for Narwhals."""

import functools
import warnings
from collections import defaultdict

import narwhals.stable.v1 as nw

from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import _is_lazy, _materialize
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)



class NarwhalsSchemaBackend(BaseSchemaBackend):
    """Base schema backend for Narwhals-backed DataFrames.

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

        Never materializes check_obj — delegates directly to .head()/.tail()
        so the result stays lazy (nw.LazyFrame) for Polars inputs.

        :param head: Number of rows to take from the head.
        :param tail: Number of rows to take from the tail.
        :param sample: Not supported — raises NotImplementedError.
        :param random_state: Ignored (no random sampling supported).
        :raises NotImplementedError: If sample is not None, or if tail= is
            requested on a SQL-lazy backend (ibis.Table) that does not support
            TAIL without forced full ordering.
        """
        if sample is not None:
            raise NotImplementedError(
                "sample= is not supported in the Narwhals backend. "
                "Use head= or tail= instead."
            )

        if head is None and tail is None:
            return check_obj

        # Guard: SQL-lazy backends don't support tail without full ordering
        if tail is not None:
            native = nw.to_native(check_obj)
            if hasattr(native, "execute"):  # ibis.Table has .execute(); pl.LazyFrame does not
                raise NotImplementedError(
                    "tail= is not supported on SQL-lazy backends (Ibis, DuckDB, PySpark) "
                    "because SQL has no native TAIL without forced full ordering. "
                    "Use head= instead."
                )

        obj_subsample = []
        if head is not None:
            obj_subsample.append(check_obj.head(head))   # lazy — no _materialize()
        if tail is not None:
            obj_subsample.append(check_obj.tail(tail))   # lazy — polars-only (guarded above)

        return nw.concat(obj_subsample).unique()

    def run_check(self, check_obj, schema, check, check_index, *args):
        """Execute a single Check object and return a CoreCheckResult.

        Single unified code path — no _is_ibis_result bifurcation.
        Materializes only the scalar passed bool via _materialize(check_passed).
        failure_cases and check_output stay as Narwhals wrappers in the returned
        CoreCheckResult; callers (failure_cases_metadata) materialize as needed.
        """
        check_result = check(check_obj, *args)

        passed_lf = check_result.check_passed  # nw.LazyFrame or nw.DataFrame
        passed = bool(_materialize(passed_lf)[CHECK_OUTPUT_KEY][0])

        message = None
        failure_cases = None

        if not passed:
            if check_result.failure_cases is None:
                # Expr path: postprocess_expr_output deferred failure_cases computation.
                # Reconstruct from the stored nw.Expr and the original check_obj frame.
                if isinstance(check_result.check_output, nw.Expr):
                    frame = nw.from_native(check_obj, eager_or_interchange_only=False)
                    expr = check_result.check_output
                    check_col = frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))
                    if check.ignore_na:
                        check_col = check_col.with_columns(
                            nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
                        )
                    fc = check_col.filter(~nw.col(CHECK_OUTPUT_KEY))
                    if check_result.checked_object is not None:
                        key = check_result.checked_object.key
                        if key and key != "*":
                            fc = fc.select(key)
                        else:
                            fc = fc.drop(CHECK_OUTPUT_KEY)
                    if check.n_failure_cases is not None:
                        fc = fc.head(check.n_failure_cases)
                    failure_cases = fc
                else:
                    failure_cases = passed
                message = f"Check '{check}' failed."
            else:
                fc = check_result.failure_cases
                # Drop CHECK_OUTPUT_KEY column if present (wide table includes it for key=="*" checks)
                if CHECK_OUTPUT_KEY in fc.collect_schema().names():
                    fc = fc.drop(CHECK_OUTPUT_KEY)
                failure_cases = fc  # Narwhals wrapper — NOT collected here
                message = f"Check '{check}' failed."

            if check.raise_warning:
                warnings.warn(message, SchemaWarning)
                return CoreCheckResult(
                    passed=True,
                    check=check,
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                )

        return CoreCheckResult(
            passed=passed,
            check=check,
            check_index=check_index,
            check_output=check_result.check_output,  # stays lazy — NOT _materialize() here
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            message=message,
            failure_cases=failure_cases,             # Narwhals wrapper — NOT _to_native() here
        )

    def is_float_dtype(self, check_obj, col_name: str) -> bool:
        """Return True if the column col_name has a float dtype.

        Uses collect_schema() so it works on both LazyFrame and DataFrame
        without triggering full materialization.

        :param check_obj: Narwhals LazyFrame or DataFrame.
        :param col_name: Name of the column to inspect.
        :returns: True if the column dtype is a floating-point type.
        """
        return check_obj.collect_schema()[col_name].is_float()

    def failure_cases_metadata(
        self,
        schema_name: str,
        schema_errors: list[SchemaError],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception.

        Backend-agnostic: returns native ibis.Table for ibis inputs and
        pl.LazyFrame/pl.DataFrame for polars inputs — no forced polars
        conversion, no Arrow roundtrip for lazy/SQL backends.
        """
        error_counts: dict[str, int] = defaultdict(int)
        failure_case_collection = []

        for err in schema_errors:
            error_counts[err.reason_code] += 1

            check_identifier = (
                None
                if err.check is None
                else (
                    err.check
                    if isinstance(err.check, str)
                    else (
                        err.check.error
                        if err.check.error is not None
                        else (
                            err.check.name
                            if err.check.name is not None
                            else str(err.check)
                        )
                    )
                )
            )

            # Wrap any native frame (pl.DataFrame, pl.LazyFrame, ibis.Table) back to Narwhals
            # so the type checks below work uniformly.
            # Python scalars/None/bool raise TypeError — leave fc unchanged (scalar path below).
            fc = err.failure_cases
            try:
                fc = nw.from_native(fc, eager_or_interchange_only=False)
            except TypeError:
                pass

            if isinstance(fc, (nw.LazyFrame, nw.DataFrame)) and _is_lazy(fc):
                # --- Lazy/SQL path (polars-lazy nw.LazyFrame or ibis nw.DataFrame) ---
                # Use Narwhals ops only — no Arrow roundtrip, no polars import in this path.
                # Row index is always None — no forced materialization for ordering.
                col_names = fc.collect_schema().names()

                if len(col_names) == 1:
                    # Single-column: rename directly to "failure_case"
                    enriched = fc.rename({col_names[0]: "failure_case"})
                else:
                    # Multi-column: build a readable "col=value, col=value" string per row.
                    # nw.concat_str() is cross-backend (polars and ibis) and stays lazy.
                    parts = [
                        nw.lit(f"{c}=").cast(nw.String) + nw.col(c).cast(nw.String)
                        for c in col_names
                    ]
                    enriched = fc.select(nw.concat_str(*parts, separator=", ").alias("failure_case"))

                enriched = enriched.with_columns(
                    nw.lit(err.schema.__class__.__name__).alias("schema_context"),
                    nw.lit(err.schema.name).alias("column"),
                    nw.lit(check_identifier).alias("check"),
                    nw.lit(err.check_index).cast(nw.Int32).alias("check_number"),
                    nw.lit(None).cast(nw.Int32).alias("index"),
                )
                failure_case_collection.append(nw.to_native(enriched))

            elif isinstance(fc, (nw.LazyFrame, nw.DataFrame)):
                # --- Eager polars path (nw.DataFrame wrapping pl.DataFrame) ---
                # Keep existing polars-based logic — works correctly for eager inputs.
                # Row index is derivable from check_output.
                # This branch is only reached for eager polars DataFrames, so polars is present.
                try:
                    import polars as pl
                except ImportError:
                    pl = None  # type: ignore[assignment]
                fc_eager = _materialize(fc)
                pl_fc = pl.from_arrow(fc_eager.to_arrow())

                # Compute row indices of failing cases from check_output.
                resolved_co = None
                if err.check_output is not None:
                    co = err.check_output
                    if isinstance(co, (nw.LazyFrame, nw.DataFrame)):
                        resolved_co = co
                    elif not isinstance(co, nw.Expr):
                        resolved_co = nw.from_native(co, eager_or_interchange_only=False)
                    # nw.Expr: resolved_co stays None (err.data unavailable)

                if resolved_co is not None:
                    co_eager = _materialize(resolved_co)
                    try:
                        co_indexed = co_eager.with_row_index("index")
                    except Exception:
                        co_indexed = co_eager.with_row_count("index")
                    failing_indices = co_indexed.filter(
                        ~nw.col(CHECK_OUTPUT_KEY)
                    )["index"].to_list()
                    index = pl.Series("index", failing_indices, dtype=pl.Int32)
                else:
                    index = pl.Series("index", [None] * len(pl_fc), dtype=pl.Int32)

                if len(pl_fc.columns) > 1:
                    failure_cases_df = pl_fc.with_columns(
                        failure_case=pl.Series(pl_fc.rows(named=True))
                    ).select(pl.col.failure_case.struct.json_encode())
                else:
                    failure_cases_df = pl_fc.rename(
                        {pl_fc.columns[0]: "failure_case"}
                    )

                failure_cases_df = failure_cases_df.with_columns(
                    schema_context=pl.lit(err.schema.__class__.__name__),
                    column=pl.lit(err.schema.name),
                    check=pl.lit(check_identifier),
                    check_number=pl.lit(err.check_index),
                    index=index.limit(failure_cases_df.shape[0]),
                ).cast(
                    {
                        "failure_case": pl.Utf8,
                        "column": pl.String,
                        "index": pl.Int32,
                        "check_number": pl.Int32,
                    }
                )
                failure_case_collection.append(failure_cases_df)

            else:
                # --- Scalar path (Python scalars, strings, etc.) ---
                try:
                    import polars as pl
                except ImportError:
                    pl = None  # type: ignore[assignment]
                scalar_failure_cases = defaultdict(list)
                scalar_failure_cases["failure_case"].append(err.failure_cases)
                scalar_failure_cases["schema_context"].append(
                    err.schema.__class__.__name__
                )
                scalar_failure_cases["column"].append(err.schema.name)
                scalar_failure_cases["check"].append(check_identifier)
                scalar_failure_cases["check_number"].append(err.check_index)
                scalar_failure_cases["index"].append(None)
                failure_cases_df = pl.DataFrame(scalar_failure_cases).cast(
                    {
                        "check_number": pl.Int32,
                        "column": pl.String,
                        "index": pl.Int32,
                    }
                )
                failure_case_collection.append(failure_cases_df)

        # Backend-aware concat: ibis uses .union(), polars uses pl.concat().
        if failure_case_collection:
            first = failure_case_collection[0]
            if hasattr(first, "union"):  # ibis.Table
                failure_cases = functools.reduce(lambda a, b: a.union(b), failure_case_collection)
            else:
                # All items are pl.DataFrame or pl.LazyFrame — polars is present here.
                try:
                    import polars as pl
                except ImportError:
                    pl = None  # type: ignore[assignment]
                failure_cases = pl.concat(failure_case_collection)  # pl.LazyFrame or pl.DataFrame
        else:
            try:
                import polars as pl
            except ImportError:
                pl = None  # type: ignore[assignment]
            failure_cases = pl.DataFrame() if pl is not None else None

        error_handler = ErrorHandler()
        # Only collect errors with a valid reason_code; errors without one
        # (e.g. manually-constructed SchemaError stubs) are silently skipped.
        valid_errors = [e for e in schema_errors if e.reason_code is not None]
        error_handler.collect_errors(valid_errors)
        error_dicts = {}

        def defaultdict_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: defaultdict_to_dict(v) for k, v in d.items()}
            return d

        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema_name)
            error_dicts = defaultdict_to_dict(error_dicts)

        error_counts = defaultdict(int)  # type: ignore
        for error in error_handler.collected_errors:
            error_counts[error["reason_code"].name] += 1

        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=error_dicts,
            error_counts=error_counts,
        )

    def drop_invalid_rows(self, check_obj, error_handler):
        """Remove invalid rows — pure Narwhals, no backend delegation.

        Builds a pass-mask boolean column per check_output, combines with
        nw.all_horizontal, filters, and drops the temporary columns.
        Works identically for polars lazy frames and ibis tables.

        Two check_output conventions are handled:
        - nw.Expr (DATAFRAME_CHECK path): True=row passes check
        - nw.LazyFrame/DataFrame with CHECK_OUTPUT_KEY=True meaning "failed"
          (SERIES_CONTAINS_NULLS / check_nullable): True=row has null (failing).
          Reconstructed as ~col.is_null() from err.schema.selector.

        :param check_obj: The frame to filter.
        :param error_handler: ErrorHandler whose schema_errors carry check_output.
        :returns: Filtered frame with only rows where all checks passed.
        """
        errors = getattr(error_handler, "schema_errors", [])
        if not errors:
            return check_obj

        # Collect (col_name, pass_expr) pairs where pass_expr returns True for valid rows.
        pass_exprs = []
        for i, err in enumerate(errors):
            co = err.check_output
            col_name = f"__check_output_{i}__"

            if isinstance(co, nw.Expr):
                # DATAFRAME_CHECK path: True=pass. Apply ignore_na is handled later.
                pass_exprs.append((col_name, co, err.check))
            elif (
                isinstance(co, (nw.LazyFrame, nw.DataFrame))
                and CHECK_OUTPUT_KEY in co.collect_schema().names()
                and err.reason_code == SchemaErrorReason.SERIES_CONTAINS_NULLS
                and err.schema is not None
                and hasattr(err.schema, "selector")
            ):
                # check_nullable path: True=null (failing). Reconstruct as ~is_null().
                selector = err.schema.selector
                not_null_expr = ~nw.col(selector).is_null()
                pass_exprs.append((col_name, not_null_expr, None))

        if not pass_exprs:
            return check_obj

        frame = nw.from_native(check_obj, eager_or_interchange_only=False)
        bool_cols = [col_name for col_name, _, _ in pass_exprs]

        # Build wide frame: single with_columns call for all exprs.
        wide = frame.with_columns([
            expr.alias(col_name)
            for col_name, expr, _ in pass_exprs
        ])

        # Apply ignore_na at column level for expr-based checks (avoids ibis SQL issues).
        ignore_na_cols = [
            col_name
            for col_name, _, check in pass_exprs
            if check is not None and getattr(check, "ignore_na", False)
        ]
        if ignore_na_cols:
            wide = wide.with_columns([
                (nw.col(c) | nw.col(c).is_null()).alias(c)
                for c in ignore_na_cols
            ])

        filtered = wide.filter(nw.all_horizontal(*[nw.col(c) for c in bool_cols]))
        result = filtered.drop(bool_cols)

        # Preserve input type: native in -> native out, Narwhals in -> Narwhals out
        if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)):
            return result
        return nw.to_native(result)
