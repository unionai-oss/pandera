"""Base schema backend for Narwhals."""

import functools
import warnings
from collections import defaultdict
from typing import Any

import narwhals.stable.v1 as nw

from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import _is_lazy, _is_sql_lazy, _materialize
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)

try:
    import polars as pl  # noqa: F401  # used in eager/scalar failure_cases paths
except ImportError:  # pragma: no cover — polars is optional
    pl = None  # type: ignore[assignment]


def _check_identifier(err: SchemaError) -> Any:
    """Derive a short, human-readable identifier for the Check on an error."""
    if err.check is None:
        return None
    if isinstance(err.check, str):
        return err.check
    if err.check.error is not None:
        return err.check.error
    if err.check.name is not None:
        return err.check.name
    return str(err.check)


def _concat_failure_cases(items: list) -> Any:
    """Concatenate per-error failure-case frames into a single frame.

    Items are one of:
    - ``nw.DataFrame`` / ``nw.LazyFrame`` — from ``_build_lazy_failure_case``
      (Polars LazyFrame, Ibis, PySpark). Dispatch on ``item.implementation``.
    - ``pl.DataFrame`` — from ``_build_eager_failure_case`` and
      ``_build_scalar_failure_case`` (eager Polars path).

    For PySpark-backed Narwhals frames: unwrap to native PySpark DataFrames
    and union via ``pyspark.sql.DataFrame.union()``. Scalar ``pl.DataFrame``
    items from ``_build_scalar_failure_case`` cannot be converted to PySpark
    without a SparkSession — they are skipped for the PySpark path and a
    ``SchemaWarning`` is emitted naming the affected columns.
    For Ibis-backed Narwhals frames: unwrap to native ibis Tables and union
    via ``ibis.Table.union()``.
    For Polars-backed Narwhals LazyFrame: stays lazy when only narwhals items
    are present; collects and merges eager ``pl.DataFrame`` items (from
    ``_build_eager_failure_case`` / ``_build_scalar_failure_case``) when both
    are present — both sources can coexist in a single polars validation run.
    For native ``pl.DataFrame`` items: ``pl.concat``.
    Returns an empty ``pl.DataFrame`` if the collection is empty.
    """
    if not items:
        return pl.DataFrame() if pl is not None else None  # pragma: no cover

    # Separate Narwhals-wrapped items from native Polars items
    nw_items = [
        item
        for item in items
        if isinstance(item, (nw.DataFrame, nw.LazyFrame))
    ]
    pl_items = [
        item
        for item in items
        if not isinstance(item, (nw.DataFrame, nw.LazyFrame))
    ]

    if nw_items:
        first_nw = nw_items[0]
        if first_nw.implementation in (
            nw.Implementation.PYSPARK,
            nw.Implementation.PYSPARK_CONNECT,
        ):
            # PySpark path: unwrap to native PySpark DataFrames and union.
            # Scalar Polars items (from _build_scalar_failure_case) cannot be
            # converted to PySpark without a SparkSession — they are skipped,
            # but a SchemaWarning is emitted so users know about the loss.
            if pl_items:
                dropped_info = []
                for item in pl_items:
                    if (
                        isinstance(item, pl.DataFrame)
                        and "column" in item.columns
                    ):
                        dropped_info.extend(item["column"].to_list())
                if dropped_info:
                    warnings.warn(
                        "Some schema-level failure cases (columns: "
                        + repr(dropped_info)
                        + ") could not be included in the PySpark failure_cases "
                        "output because scalar Polars frames cannot be converted "
                        "to PySpark without a live SparkSession. These schema "
                        "errors are still reported in SchemaErrors but their "
                        "failure_cases rows are omitted from the combined frame. "
                        "This gap is tracked for a future release.",
                        SchemaWarning,
                        stacklevel=6,
                    )
            native_items = [nw.to_native(item) for item in nw_items]
            return functools.reduce(lambda a, b: a.union(b), native_items)
        elif first_nw.implementation == nw.Implementation.POLARS:
            # Polars lazy path: use nw.concat to stay lazy, then unwrap.
            # When pl_items are also present (schema-level failure cases from
            # _build_eager_failure_case / _build_scalar_failure_case producing
            # pl.DataFrame alongside data-check failure cases from
            # _build_lazy_failure_case producing nw.LazyFrame), collect the
            # lazy result and concatenate via pl.concat. Polars has no
            # SparkSession barrier — both sources merge cleanly, so no
            # SchemaWarning is needed (unlike the PySpark branch which
            # warns-and-drops because it cannot create a SparkSession).
            nw_types = {type(i) for i in nw_items}
            if len(nw_types) > 1:  # pragma: no cover
                raise ValueError(
                    "nw_items must be homogeneous (all LazyFrame or all DataFrame); "
                    f"got types: {[type(i).__name__ for i in nw_items]}"
                )
            lazy_result = nw.to_native(nw.concat(nw_items))
            if pl_items:
                eager_result = (
                    lazy_result.collect()
                    if isinstance(lazy_result, pl.LazyFrame)
                    else lazy_result
                )
                return pl.concat([eager_result] + pl_items)
            return lazy_result
        else:
            # SQL-lazy path (Ibis, DuckDB, etc.): unwrap to native and union.
            native_items = [nw.to_native(item) for item in nw_items]
            return functools.reduce(lambda a, b: a.union(b), native_items)

    # All-Polars path: pl.DataFrame items from eager/scalar builders
    return pl.concat(pl_items) if pl is not None else None  # pragma: no cover


class NarwhalsSchemaBackend(BaseSchemaBackend):
    """Base schema backend for Narwhals-backed DataFrames.

    Provides shared helpers used by ColumnBackend (components.py) and
    DataFrameSchemaBackend (container.py).
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

        Stays lazy whenever possible. ``head=``/``tail=`` delegate to
        ``nw.LazyFrame.head()/tail()`` — no ``_materialize()`` call.
        ``sample=`` is only supported for eager frames (``nw.DataFrame``
        wrapping a ``pl.DataFrame``); it requires ``nw.DataFrame.sample``
        which is not available on ``nw.LazyFrame`` or SQL-lazy backends.

        :param head: Number of rows to take from the head.
        :param tail: Number of rows to take from the tail.
        :param sample: Number of rows to randomly sample (eager polars only).
        :param random_state: Seed forwarded to ``sample(seed=...)``.
        :raises NotImplementedError: If ``sample=`` is requested on a lazy or
            SQL-lazy input, or if ``tail=`` is requested on a SQL-lazy backend
            (ibis.Table) that does not support TAIL without forced full
            ordering.
        """
        if head is None and tail is None and sample is None:
            return check_obj

        if tail is not None and _is_sql_lazy(check_obj):
            raise NotImplementedError(
                "tail= is not supported on SQL-lazy backends (Ibis, DuckDB, "
                "PySpark) because SQL has no native TAIL without forced full "
                "ordering. Use head= instead."
            )

        if sample is not None:
            # Sampling is not representable as a pure expression on lazy or
            # SQL-lazy frames, so we restrict it to eager polars inputs. This
            # matches the behaviour of the legacy Polars backend.
            if isinstance(check_obj, nw.LazyFrame):
                raise NotImplementedError(
                    "sample= is not supported for lazy frames (polars "
                    "LazyFrame). Use head=/tail= or call .collect() on the "
                    "input before validation."
                )
            if _is_sql_lazy(check_obj):
                raise NotImplementedError(
                    "sample= is not supported on SQL-lazy backends (Ibis, "
                    "DuckDB, PySpark). Use head= instead."
                )

        obj_subsample = []
        if head is not None:
            obj_subsample.append(check_obj.head(head))
        if tail is not None:
            obj_subsample.append(check_obj.tail(tail))
        if sample is not None:
            obj_subsample.append(check_obj.sample(n=sample, seed=random_state))

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
                    frame = nw.from_native(
                        check_obj, eager_or_interchange_only=False
                    )
                    expr = check_result.check_output
                    check_col = frame.with_columns(
                        expr.alias(CHECK_OUTPUT_KEY)
                    )
                    if check.ignore_na:
                        check_col = check_col.with_columns(
                            nw.col(CHECK_OUTPUT_KEY)
                            | nw.col(CHECK_OUTPUT_KEY).is_null()
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
            failure_cases=failure_cases,  # Narwhals wrapper — NOT _to_native() here
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
        failure_case_collection: list = []

        for err in schema_errors:
            check_identifier = _check_identifier(err)

            # Wrap native frames (pl.DataFrame, pl.LazyFrame, ibis.Table) as
            # Narwhals wrappers for uniform dispatch. Python scalars raise
            # TypeError — handled by the scalar path below.
            fc = err.failure_cases
            try:
                fc = nw.from_native(fc, eager_or_interchange_only=False)
            except TypeError:
                pass

            if isinstance(fc, (nw.LazyFrame, nw.DataFrame)) and _is_lazy(fc):
                failure_case_collection.append(
                    self._build_lazy_failure_case(fc, err, check_identifier)
                )
            elif isinstance(fc, (nw.LazyFrame, nw.DataFrame)):
                failure_case_collection.append(
                    self._build_eager_failure_case(fc, err, check_identifier)
                )
            else:
                failure_case_collection.append(
                    self._build_scalar_failure_case(err, check_identifier)
                )

        failure_cases = _concat_failure_cases(failure_case_collection)

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

    @staticmethod
    def _build_lazy_failure_case(fc, err: SchemaError, check_identifier):
        """Build a lazy/SQL failure-case frame using Narwhals ops only.

        Works uniformly for ``polars.LazyFrame`` and ``ibis.Table`` — no
        Arrow roundtrip, no polars import. Row index is always ``None``
        since SQL has no natural row ordering.

        Returns a narwhals-wrapped frame (not a native frame) so that
        ``_concat_failure_cases`` can dispatch on ``item.implementation``
        instead of module-string sniffing.
        """
        col_names = fc.collect_schema().names()
        if len(col_names) == 1:
            enriched = fc.rename({col_names[0]: "failure_case"})
        else:
            parts = [
                nw.lit(f"{c}=").cast(nw.String) + nw.col(c).cast(nw.String)
                for c in col_names
            ]
            enriched = fc.select(
                nw.concat_str(*parts, separator=", ").alias("failure_case")
            )

        enriched = enriched.with_columns(
            nw.lit(err.schema.__class__.__name__).alias("schema_context"),
            nw.lit(err.schema.name).alias("column"),
            nw.lit(check_identifier).alias("check"),
            nw.lit(err.check_index).cast(nw.Int32).alias("check_number"),
            nw.lit(None).cast(nw.Int32).alias("index"),
        )
        # Return narwhals-wrapped frame — _concat_failure_cases dispatches on
        # item.implementation to handle PySpark vs ibis vs polars without
        # module-string sniffing.
        return enriched

    @staticmethod
    def _build_eager_failure_case(fc, err: SchemaError, check_identifier):
        """Build an eager polars failure-case frame with row-index enrichment.

        Only reached for eager polars DataFrames; ``polars`` is guaranteed
        to be importable here.
        """
        assert pl is not None, "polars is required for eager failure_cases"
        fc_eager = _materialize(fc)
        pl_fc = pl.from_arrow(fc_eager.to_arrow())

        resolved_co = None
        if err.check_output is not None:
            co = err.check_output
            if isinstance(co, (nw.LazyFrame, nw.DataFrame)):
                resolved_co = co
            elif not isinstance(co, nw.Expr):
                resolved_co = nw.from_native(
                    co, eager_or_interchange_only=False
                )

        if resolved_co is not None:
            co_eager = _materialize(resolved_co)
            try:
                co_indexed = co_eager.with_row_index("index")
            except AttributeError:
                # Older polars: ``with_row_index`` was called ``with_row_count``.
                co_indexed = co_eager.with_row_count("index")
            failing_indices = co_indexed.filter(~nw.col(CHECK_OUTPUT_KEY))[
                "index"
            ].to_list()
            index = pl.Series("index", failing_indices, dtype=pl.Int32)
        else:
            index = pl.Series("index", [None] * len(pl_fc), dtype=pl.Int32)

        assert isinstance(pl_fc, pl.DataFrame)
        if len(pl_fc.columns) > 1:
            failure_cases_df = pl_fc.with_columns(
                failure_case=pl.Series(pl_fc.rows(named=True))
            ).select(pl.col.failure_case.struct.json_encode())
        else:
            failure_cases_df = pl_fc.rename({pl_fc.columns[0]: "failure_case"})

        return failure_cases_df.with_columns(
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

    @staticmethod
    def _build_scalar_failure_case(err: SchemaError, check_identifier):
        """Build a failure-case frame for Python scalars/strings/None."""
        assert pl is not None, "polars is required for scalar failure_cases"
        scalar_failure_cases: dict = defaultdict(list)
        scalar_failure_cases["failure_case"].append(err.failure_cases)
        scalar_failure_cases["schema_context"].append(
            err.schema.__class__.__name__
        )
        scalar_failure_cases["column"].append(err.schema.name)
        scalar_failure_cases["check"].append(check_identifier)
        scalar_failure_cases["check_number"].append(err.check_index)
        scalar_failure_cases["index"].append(None)
        return pl.DataFrame(scalar_failure_cases).cast(
            {
                "check_number": pl.Int32,
                "column": pl.String,
                "index": pl.Int32,
            }
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
        wide = frame.with_columns(
            [expr.alias(col_name) for col_name, expr, _ in pass_exprs]
        )

        # Apply ignore_na at column level for expr-based checks (avoids ibis SQL issues).
        ignore_na_cols = [
            col_name
            for col_name, _, check in pass_exprs
            if check is not None and getattr(check, "ignore_na", False)
        ]
        if ignore_na_cols:
            wide = wide.with_columns(
                [
                    (nw.col(c) | nw.col(c).is_null()).alias(c)
                    for c in ignore_na_cols
                ]
            )

        filtered = wide.filter(
            nw.all_horizontal(*[nw.col(c) for c in bool_cols])
        )
        result = filtered.drop(bool_cols)

        # Preserve input type: native in -> native out, Narwhals in -> Narwhals out
        if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)):
            return result
        return nw.to_native(result)
