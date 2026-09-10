"""Base schema backend for Narwhals."""

import functools
import warnings
from collections import defaultdict
from typing import Any

import narwhals.stable.v1 as nw

from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import (
    _EAGER_PANDAS_LIKE_IMPLEMENTATIONS,
    _is_lazy,
    _is_pandas_like,
    _is_sql_lazy,
    _materialize,
)
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.narwhals.checks import (
    NarwhalsCheckBackend,
    _use_input_nullness,
)
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

try:
    import pyarrow as _pa  # used in the pyarrow failure_cases paths
except ImportError:  # pragma: no cover — pyarrow is optional
    _pa = None  # type: ignore[assignment]


def _errors_implementation(schema_errors: list[SchemaError]) -> Any:
    """Infer the frame backend a batch of errors came from.

    Scalar failure cases (missing column, wrong dtype, …) carry no frame, so
    their builder has nothing to dispatch on. Taking the implementation from
    whichever errors *do* carry a frame keeps every piece of one batch on the
    same backend, which is what ``_concat_failure_cases`` needs to combine
    them. Returns ``None`` when no error in the batch carries a frame.
    """
    for err in schema_errors:
        try:
            fc = nw.from_native(
                err.failure_cases, eager_or_interchange_only=False
            )
        except TypeError:
            continue
        if isinstance(fc, (nw.DataFrame, nw.LazyFrame)):
            return fc.implementation
    return None


def _lit_nullable_int32(value, implementation) -> Any:
    """Int32 literal that tolerates ``None`` on pandas-like backends.

    ``nw.lit(None).cast(nw.Int32)`` raises on pandas-like implementations
    (numpy cannot astype ``None`` to int32). Fall back to a Float64 cast so
    the column materializes as float64 with NaN — the idiomatic pandas
    representation of a missing integer.
    """
    if value is None and implementation in _EAGER_PANDAS_LIKE_IMPLEMENTATIONS:
        return nw.lit(None).cast(nw.Float64)
    return nw.lit(value).cast(nw.Int32)


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


def _concat_failure_cases(items: list, implementation=None) -> Any:
    """Concatenate per-error failure-case frames into a single frame.

    Items are one of:
    - ``nw.DataFrame`` / ``nw.LazyFrame`` — from ``_build_lazy_failure_case``
      (Polars LazyFrame, Ibis, PySpark, pandas). Dispatch on
      ``item.implementation``.
    - ``pl.DataFrame`` — from ``_build_eager_failure_case`` and
      ``_build_scalar_failure_case`` (eager Polars path).

    ``implementation`` is the ``nw.Implementation`` of the frame that was
    validated (when known). It disambiguates the all-scalar case — when every
    failure case is a Python scalar there is no narwhals-wrapped item to
    dispatch on, and pandas-like validations must still return a pandas
    failure-cases frame instead of the historical polars default.

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
    For pandas-like Narwhals frames (pandas, modin, cuDF): unwrap to native
    frames and concatenate via ``pandas.concat``; scalar items are converted
    to pandas frames.
    For native ``pl.DataFrame`` items: ``pl.concat``.
    Returns an empty ``pl.DataFrame`` (or ``pandas.DataFrame`` for
    pandas-like validations) if the collection is empty.
    """
    is_pandas_like = implementation in _EAGER_PANDAS_LIKE_IMPLEMENTATIONS

    if not items:
        if is_pandas_like:  # pragma: no cover — defensive default
            import pandas as pd

            return pd.DataFrame()
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
        elif first_nw.implementation == nw.Implementation.PYARROW:
            # pyarrow.Table has no ``.union()``; concatenate via narwhals and
            # hand back the native table. Every piece of a pyarrow run is
            # built by the pyarrow builders above, so there are no pl_items
            # to merge here.
            return nw.to_native(nw.concat(nw_items))
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
        elif first_nw.implementation in _EAGER_PANDAS_LIKE_IMPLEMENTATIONS:
            # pandas-like path (pandas, modin, cuDF): materialize the
            # Narwhals items to native frames and concatenate with pandas.
            # Scalar failure-case items (pl_items, built by
            # _build_scalar_failure_case) are converted via to_dict — no
            # Arrow roundtrip, so pyarrow is not required.
            import pandas as pd  # local import: module must stay pandas-free

            native_items = [
                nw.to_native(_materialize(item)) for item in nw_items
            ]
            for item in pl_items:
                if pl is not None and isinstance(item, pl.DataFrame):
                    item = pd.DataFrame(item.to_dict(as_series=False))
                native_items.append(item)
            return pd.concat(native_items, ignore_index=True)
        else:
            # SQL-lazy path (Ibis, DuckDB, etc.): unwrap to native and union.
            native_items = [nw.to_native(item) for item in nw_items]
            return functools.reduce(lambda a, b: a.union(b), native_items)

    if is_pandas_like:
        # All-scalar pandas-like path: every failure case was a Python
        # scalar (e.g. wrong-dtype or missing-column errors), so there is no
        # narwhals-wrapped item to dispatch on — build a pandas frame.
        import pandas as pd  # local import: module must stay pandas-free

        converted = [
            (
                pd.DataFrame(item.to_dict(as_series=False))
                if pl is not None and isinstance(item, pl.DataFrame)
                else item
            )
            for item in pl_items
        ]
        return pd.concat(converted, ignore_index=True)

    # All-Polars path: pl.DataFrame items from eager/scalar builders
    return pl.concat(pl_items) if pl is not None else None  # pragma: no cover


class NarwhalsSchemaBackend(BaseSchemaBackend):
    """Base schema backend for Narwhals-backed DataFrames.

    Provides shared helpers used by ColumnBackend (components.py) and
    DataFrameSchemaBackend (container.py).
    """

    @staticmethod
    def is_native_delegated_check(check) -> bool:
        """True if a check must run through the native pandas check backend.

        ``Hypothesis`` checks rely on scipy statistical tests over grouped
        samples with no Narwhals-expression equivalent, so for eager pandas
        frames they are delegated to the native pandas hypothesis backend.
        (``groupby`` column-check-groups are handled natively by the Narwhals
        check backend — see ``NarwhalsCheckBackend.apply_groupby``.)
        """
        from pandera.api.hypotheses import Hypothesis

        return isinstance(check, Hypothesis)

    def run_native_check(
        self, check_obj, schema, check, check_index, column_name=None
    ) -> CoreCheckResult:
        """Run a single check via the native pandas check backend.

        Used for ``Hypothesis`` checks on eager pandas frames. ``check_obj`` is
        a Narwhals frame; it is unwrapped to the native pandas frame so the
        check dispatches to the native pandas hypothesis backend (which stays
        registered for ``pd.DataFrame`` under the override). Returns the native
        ``CoreCheckResult`` unchanged — the Narwhals error-collection loop
        already understands it.
        """
        from pandera.api.narwhals.utils import _to_native
        from pandera.backends.pandas.base import PandasSchemaBackend

        native_obj = _to_native(check_obj)
        args = () if column_name is None else (column_name,)
        # run_check does not rely on backend instance state, so a bare native
        # backend instance is enough to reuse its implementation.
        return PandasSchemaBackend().run_check(
            native_obj, schema, check, check_index, *args
        )

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
        if tail is not None and _is_sql_lazy(check_obj):
            raise NotImplementedError(
                "tail= is not supported on SQL-lazy backends (Ibis, DuckDB, PySpark) "
                "because SQL has no native TAIL without forced full ordering. "
                "Use head= instead."
            )

        obj_subsample = []
        if head is not None:
            obj_subsample.append(
                check_obj.head(head)
            )  # lazy — no _materialize()
        if tail is not None:
            obj_subsample.append(
                check_obj.tail(tail)
            )  # lazy — polars-only (guarded above)

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
                        na_pass = (
                            nw.col(CHECK_OUTPUT_KEY)
                            | nw.col(CHECK_OUTPUT_KEY).is_null()
                        )
                        checked_key = (
                            check_result.checked_object.key
                            if check_result.checked_object is not None
                            else None
                        )
                        if _use_input_nullness(
                            check.ignore_na, check_col, checked_key
                        ):
                            # pandas-like: NaN comparisons yield False, not
                            # null — OR in the input column's nullness.
                            na_pass = na_pass | nw.col(checked_key).is_null()
                        check_col = check_col.with_columns(
                            na_pass.alias(CHECK_OUTPUT_KEY)
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
        errors_implementation = _errors_implementation(schema_errors)

        # Implementation of the validated frame — used to disambiguate the
        # all-scalar failure-cases path in _concat_failure_cases (e.g. pandas
        # validations must return pandas frames even when every failure case
        # is a Python scalar). ``err.data`` is nulled by the lazy
        # ErrorHandler to avoid holding frame copies, so detection falls
        # back to the schema class: pandas-API schemas are only routed to
        # this backend for pandas frames.
        data_implementation = None
        for err in schema_errors:
            data = getattr(err, "data", None)
            if data is not None:
                if not isinstance(data, (nw.DataFrame, nw.LazyFrame)):
                    try:
                        data = nw.from_native(
                            data, eager_or_interchange_only=False
                        )
                    except TypeError:
                        data = None
                if data is not None:
                    data_implementation = data.implementation
                    break
            schema_mod = (
                type(err.schema).__module__ if err.schema is not None else ""
            )
            if schema_mod.startswith(
                ("pandera.api.pandas", "pandera._pandas_deprecated")
            ):
                data_implementation = nw.Implementation.PANDAS
                break

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

            # pandas-like frames take the narwhals-native builder as well:
            # the eager builder below is a polars-specific path (Arrow
            # roundtrip into pl.DataFrame) that would convert pandas failure
            # cases to polars and leak the pandas index into the output.
            if isinstance(fc, (nw.LazyFrame, nw.DataFrame)) and (
                _is_lazy(fc) or _is_pandas_like(fc)
            ):
                failure_case_collection.append(
                    self._build_lazy_failure_case(fc, err, check_identifier)
                )
            elif isinstance(fc, (nw.LazyFrame, nw.DataFrame)):
                if fc.implementation == nw.Implementation.PYARROW:
                    failure_case_collection.append(
                        self._build_pyarrow_failure_case(
                            fc, err, check_identifier
                        )
                    )
                else:
                    failure_case_collection.append(
                        self._build_eager_failure_case(
                            fc, err, check_identifier
                        )
                    )
            elif errors_implementation == nw.Implementation.PYARROW or (
                pl is None
                and _pa is not None
                # pandas-like runs keep _build_scalar_failure_case, which
                # already builds a pandas frame when polars is absent.
                # ``errors_implementation`` is None for an all-scalar batch,
                # so the validated frame's implementation is the only usable
                # signal here; without it a pandas + pyarrow (no polars)
                # install would report failure_cases as a pyarrow.Table.
                and data_implementation
                not in _EAGER_PANDAS_LIKE_IMPLEMENTATIONS
            ):
                failure_case_collection.append(
                    self._build_pyarrow_scalar_failure_case(
                        err, check_identifier
                    )
                )
            else:
                failure_case_collection.append(
                    self._build_scalar_failure_case(err, check_identifier)
                )

        failure_cases = _concat_failure_cases(
            failure_case_collection, data_implementation
        )

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
            _lit_nullable_int32(err.check_index, fc.implementation).alias(
                "check_number"
            ),
            _lit_nullable_int32(None, fc.implementation).alias("index"),
        )
        # Return narwhals-wrapped frame — _concat_failure_cases dispatches on
        # item.implementation to handle PySpark vs ibis vs polars without
        # module-string sniffing.
        return enriched

    @staticmethod
    def _resolved_check_output(err: SchemaError):
        """Normalize ``err.check_output`` to a Narwhals frame, or ``None``."""
        if err.check_output is None:
            return None
        co = err.check_output
        if isinstance(co, (nw.LazyFrame, nw.DataFrame)):
            return co
        if isinstance(co, nw.Expr):
            return None
        return nw.from_native(co, eager_or_interchange_only=False)

    @staticmethod
    def _failing_row_indices(err: SchemaError) -> list | None:
        """Positions of the rows that failed the check, if recoverable."""
        resolved_co = NarwhalsSchemaBackend._resolved_check_output(err)
        if resolved_co is None:
            return None
        co_eager = _materialize(resolved_co)
        try:
            co_indexed = co_eager.with_row_index("index")
        except AttributeError:
            # Older polars: ``with_row_index`` was called ``with_row_count``.
            co_indexed = co_eager.with_row_count("index")
        return co_indexed.filter(~nw.col(CHECK_OUTPUT_KEY))["index"].to_list()

    @staticmethod
    def _build_pyarrow_failure_case(fc, err: SchemaError, check_identifier):
        """Build an eager pyarrow failure-case table with row-index enrichment.

        Mirrors ``_build_eager_failure_case`` but stays in pyarrow rather than
        round-tripping through polars, so that a ``pandera[pyarrow]`` install
        without polars can still report failure cases — and so the reported
        frame does not change type depending on whether polars happens to be
        installed.

        Returns a Narwhals-wrapped frame so ``_concat_failure_cases`` can
        dispatch on ``item.implementation``.
        """
        import json

        fc_native = nw.to_native(_materialize(fc))
        rows = fc_native.to_pylist()
        col_names = list(fc_native.column_names)

        if len(col_names) == 1:
            key = col_names[0]
            failure_case = [
                None if row[key] is None else str(row[key]) for row in rows
            ]
        else:
            # Match the polars path, which JSON-encodes the whole row when a
            # failure case spans multiple columns.
            failure_case = [json.dumps(row, default=str) for row in rows]

        failing = NarwhalsSchemaBackend._failing_row_indices(err)
        if failing is None:
            index: list = [None] * len(rows)
        else:
            index = list(failing[: len(rows)])
            index += [None] * (len(rows) - len(index))

        table = _pa.table(
            {
                "failure_case": _pa.array(failure_case, type=_pa.string()),
                "schema_context": _pa.array(
                    [err.schema.__class__.__name__] * len(rows),
                    type=_pa.string(),
                ),
                "column": _pa.array(
                    [err.schema.name] * len(rows), type=_pa.string()
                ),
                "check": _pa.array(
                    [
                        None
                        if check_identifier is None
                        else str(check_identifier)
                    ]
                    * len(rows),
                    type=_pa.string(),
                ),
                "check_number": _pa.array(
                    [err.check_index] * len(rows), type=_pa.int32()
                ),
                "index": _pa.array(index, type=_pa.int32()),
            }
        )
        return nw.from_native(table, eager_only=True)

    @staticmethod
    def _build_pyarrow_scalar_failure_case(err: SchemaError, check_identifier):
        """Scalar failure case as a one-row pyarrow table.

        The polars equivalent is ``_build_scalar_failure_case``; this keeps a
        pyarrow validation run on a single backend so the pieces concatenate.
        """
        failure_case = err.failure_cases
        table = _pa.table(
            {
                "failure_case": _pa.array(
                    [None if failure_case is None else str(failure_case)],
                    type=_pa.string(),
                ),
                "schema_context": _pa.array(
                    [err.schema.__class__.__name__], type=_pa.string()
                ),
                "column": _pa.array([err.schema.name], type=_pa.string()),
                "check": _pa.array(
                    [
                        None
                        if check_identifier is None
                        else str(check_identifier)
                    ],
                    type=_pa.string(),
                ),
                "check_number": _pa.array([err.check_index], type=_pa.int32()),
                "index": _pa.array([None], type=_pa.int32()),
            }
        )
        return nw.from_native(table, eager_only=True)

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
            co_indexed = co_eager.with_row_index("index")
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
        """Build a failure-case frame for Python scalars/strings/None.

        Returns a ``pl.DataFrame`` when polars is installed (the historical
        behavior; downstream ``_concat_failure_cases`` branches convert as
        needed). On polars-free installs (e.g. pandas + narwhals only) a
        ``pandas.DataFrame`` is built instead.
        """
        scalar_failure_cases: dict = defaultdict(list)
        scalar_failure_cases["failure_case"].append(err.failure_cases)
        scalar_failure_cases["schema_context"].append(
            err.schema.__class__.__name__
        )
        scalar_failure_cases["column"].append(err.schema.name)
        scalar_failure_cases["check"].append(check_identifier)
        scalar_failure_cases["check_number"].append(err.check_index)
        scalar_failure_cases["index"].append(None)
        if pl is None:
            # polars-free install (e.g. pandas + narwhals only): build the
            # scalar failure-case frame with pandas instead.
            import pandas as pd

            return pd.DataFrame(dict(scalar_failure_cases))
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

        # Collect (col_name, pass_expr, check, selector) tuples where
        # pass_expr returns True for valid rows and selector identifies the
        # checked input column (None for dataframe-level checks).
        pass_exprs = []
        for i, err in enumerate(errors):
            co = err.check_output
            col_name = f"__check_output_{i}__"
            selector = getattr(err.schema, "selector", None)

            if isinstance(co, nw.Expr):
                # DATAFRAME_CHECK path: True=pass. Apply ignore_na is handled later.
                pass_exprs.append((col_name, co, err.check, selector))
            elif (
                isinstance(co, (nw.LazyFrame, nw.DataFrame))
                and CHECK_OUTPUT_KEY in co.collect_schema().names()
                and err.reason_code == SchemaErrorReason.SERIES_CONTAINS_NULLS
                and err.schema is not None
                and hasattr(err.schema, "selector")
            ):
                # check_nullable path: True=null (failing). Reconstruct as ~is_null().
                not_null_expr = ~nw.col(selector).is_null()
                pass_exprs.append((col_name, not_null_expr, None, selector))

        if not pass_exprs:
            return check_obj

        frame = nw.from_native(check_obj, eager_or_interchange_only=False)
        bool_cols = [col_name for col_name, _, _, _ in pass_exprs]

        # Build wide frame: single with_columns call for all exprs.
        wide = frame.with_columns(
            [expr.alias(col_name) for col_name, expr, _, _ in pass_exprs]
        )

        # Apply ignore_na at column level for expr-based checks (avoids ibis SQL issues).
        ignore_na_specs = [
            (col_name, selector)
            for col_name, _, check, selector in pass_exprs
            if check is not None and getattr(check, "ignore_na", False)
        ]
        if ignore_na_specs:
            na_exprs = []
            for c, selector in ignore_na_specs:
                na_pass = nw.col(c) | nw.col(c).is_null()
                if _use_input_nullness(True, wide, selector):
                    # pandas-like: NaN comparisons yield False, not null —
                    # OR in the input column's nullness.
                    na_pass = na_pass | nw.col(selector).is_null()
                na_exprs.append(na_pass.alias(c))
            wide = wide.with_columns(na_exprs)

        filtered = wide.filter(
            nw.all_horizontal(*[nw.col(c) for c in bool_cols])
        )
        result = filtered.drop(bool_cols)

        # Preserve input type: native in -> native out, Narwhals in -> Narwhals out
        if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)):
            return result
        return nw.to_native(result)
