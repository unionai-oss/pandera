"""Validation backend for Narwhals DataFrameSchema."""

from __future__ import annotations

import copy
import functools
import re
import traceback
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import get_error_category
from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import (
    _is_pandas_like,
    _materialize,
    _to_native,
    _unwrap_failure_cases,
)

if TYPE_CHECKING:
    from pandera.api.polars.container import DataFrameSchema

from pandera.backends.base import ColumnInfo, CoreCheckResult
from pandera.backends.narwhals.base import NarwhalsSchemaBackend
from pandera.config import (
    ValidationDepth,
    ValidationScope,
    config_context,
    get_config_context,
)
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
    SchemaWarning,
)
from pandera.utils import is_regex
from pandera.validation_depth import validate_scope


def _native_is_pandas_like(obj) -> bool:
    """True if a *native* (un-wrapped) frame is a pandas-like DataFrame.

    Detects pandas / modin / cudf frames from the class module prefix so we
    can decide — before wrapping into Narwhals — whether the pandas-specific
    pre-wrap steps (custom parsers, index coercion, column-name uniqueness)
    should run. Narwhals wrapping raises on duplicate column labels, so this
    must not rely on wrapping the frame.
    """
    module = type(obj).__module__
    return module.startswith(("pandas", "modin", "cudf"))


def _narwhals_target_dtype(pandera_dtype):
    """Map a pandera ``DataType`` to a Narwhals dtype for ``cast``.

    Returns the Narwhals dtype (e.g. ``nw.Int64``) or ``None`` when the pandera
    dtype has no Narwhals equivalent (e.g. some pandas extension dtypes). A
    ``None`` result means the column is left as-is — the subsequent dtype check
    reports any mismatch. This is the documented fidelity limitation of the
    Narwhals-native coerce path.
    """
    if pandera_dtype is None:
        return None
    from pandera.engines import narwhals_engine

    for probe in (pandera_dtype, str(pandera_dtype)):
        try:
            return narwhals_engine.Engine.dtype(probe).type
        except (TypeError, ValueError, SystemError):
            continue
    return None


def _to_lazy_nw(check_obj) -> nw.LazyFrame:
    """Wrap any supported native frame as a Narwhals LazyFrame."""
    wrapped = nw.from_native(check_obj, eager_or_interchange_only=False)
    if isinstance(wrapped, nw.DataFrame):
        return wrapped.lazy()
    return wrapped  # already nw.LazyFrame


def _to_frame_kind_nw(lf: nw.LazyFrame, return_type: type):
    """Unwrap a Narwhals LazyFrame to match the original native frame type.

    If the caller originally passed an eager ``pl.DataFrame``, the
    corresponding Narwhals lazy result must be collected back into an eager
    frame so the returned type matches the input. Ibis tables and
    ``pl.LazyFrame`` inputs pass through as-is.

    The decision is driven by the Narwhals Implementation and the original
    ``return_type``, rather than attribute probing (``hasattr(..., "collect")``)
    on the native object — the latter is ambiguous because both
    ``pl.LazyFrame`` and ``pl.DataFrame`` share some API surface.
    """
    # Detect "caller passed an eager polars.DataFrame" purely from return_type
    # metadata so we don't need to import polars here. Eager polars DataFrame
    # subclasses do not define ``collect`` at the class level; the lazy class
    # does.  Everything else (ibis.Table, pl.LazyFrame) is returned as-is.
    # Both conditions are required:
    # 1. No class-level .collect → distinguishes pl.DataFrame from pl.LazyFrame
    # 2. polars module prefix → distinguishes polars from PySpark (whose module
    #    starts with 'pyspark', not 'polars')
    caller_was_eager_polars = not hasattr(
        return_type, "collect"
    ) and return_type.__module__.startswith("polars")
    native = nw.to_native(lf)
    if caller_was_eager_polars:
        # Acceptable: full-frame collect only at the final validation return
        # boundary. The caller originally passed an eager frame and expects
        # an eager result back. This is a user-visible materialization at
        # schema exit, not an internal hot-path collect.
        return native.collect()
    return native


class DataFrameSchemaBackend(NarwhalsSchemaBackend):
    def validate(
        self,
        check_obj,
        schema: DataFrameSchema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        # Capture the input type so we can return the same type
        return_type = type(check_obj)

        if inplace:
            warnings.warn("setting inplace=True will have no effect.")

        error_handler = ErrorHandler(lazy)

        is_pandas_native = _native_is_pandas_like(check_obj)

        # --- pandas-only pre-wrap steps ---------------------------------------
        # These operate on the *native* pandas frame before it is wrapped into
        # Narwhals, because Narwhals has no parser step and rejects duplicate
        # column labels at wrap time.

        # Run custom parsers (schema.parsers) on the native pandas frame
        # (self-contained — the parser fn is applied directly here, no dispatch
        # to the pandas schema backend). Narwhals preserves the resulting frame
        # through wrapping.
        if is_pandas_native and getattr(schema, "parsers", None):
            check_obj = self.run_native_parsers(check_obj, schema)

        # Index/MultiIndex coercion is the one delegation to the native pandas
        # backend (Narwhals has no index concept). Coerce the index on the
        # native frame pre-wrap so the coerced index propagates through
        # Narwhals (which preserves the pandas index) to the output.
        if is_pandas_native and getattr(schema, "index", None) is not None:
            check_obj = self.coerce_native_index(
                check_obj, schema, error_handler
            )

        # add_missing_columns, set_default, and column/schema coercion are
        # applied as Narwhals-native core parsers (see below) — no delegation
        # to the pandas schema backend.

        # Column-name uniqueness must be detected on the native frame: Narwhals
        # raises ``DuplicateError`` when wrapping a frame with duplicate labels,
        # so the check cannot run after wrapping.
        if is_pandas_native:
            dup_error = self.check_native_column_names_unique(
                check_obj, schema
            )
            if dup_error is not None:
                # collect_error raises immediately in non-lazy mode; in lazy
                # mode we raise below because the frame cannot be wrapped.
                error_handler.collect_error(
                    get_error_category(dup_error.reason_code),
                    dup_error.reason_code,
                    dup_error,
                )
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=check_obj,
                )

        # Convert to Narwhals LazyFrame — all parsers operate on LazyFrame
        check_lf = _to_lazy_nw(check_obj)

        # Index/MultiIndex validation is a pandas-only concept. For eager
        # pandas-like frames it is delegated to the native Index/MultiIndex
        # backends in ``run_index_checks`` (Narwhals preserves the pandas
        # index through its operations, so the parsed frame still carries it).
        # For non-pandas frames that somehow carry an index component, warn and
        # skip — Narwhals has no notion of an index there.
        if getattr(schema, "index", None) is not None and not is_pandas_native:
            warnings.warn(
                "index validation is not supported by the narwhals backend "
                "for non-pandas frames. The `index` component of this schema "
                "will not be validated.",
                SchemaWarning,
                stacklevel=5,
            )

        column_info = self.collect_column_info(check_lf, schema)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        # Core parsers, mirroring the native pandas ordering:
        # add_missing_columns → strict_filter_columns → set_defaults →
        # coerce_dtype. All are Narwhals-native (operate on the LazyFrame).
        core_parsers: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [
            (self.add_missing_columns, (schema, column_info)),
            (self.strict_filter_columns, (schema, column_info)),
            (self.set_defaults, (schema,)),
            (self.coerce_dtype, (schema,)),
        ]

        for parser, args in core_parsers:
            try:
                check_lf = parser(check_lf, *args)
            except SchemaError as exc:
                error_handler.collect_error(
                    get_error_category(exc.reason_code),
                    exc.reason_code,
                    exc,
                )
            except SchemaErrors as exc:
                error_handler.collect_errors(exc.schema_errors)

        # add_missing_columns may have added columns — regenerate column info
        # so downstream presence checks and components see the current frame.
        column_info = self.collect_column_info(check_lf, schema)

        # collect schema components
        components = self.collect_schema_components(
            check_lf, schema, column_info
        )

        # subsample on the Narwhals LazyFrame — no native round-trip before checks
        sample_obj = self.subsample(
            check_lf,
            head,
            tail,
            sample,
            random_state,
        )
        # subsample() returns nw.LazyFrame (unchanged) or nw.DataFrame (if head/tail used);
        # normalize to LazyFrame for uniform check execution
        if isinstance(sample_obj, nw.DataFrame):
            sample_lf = sample_obj.lazy()
        else:
            sample_lf = sample_obj  # already nw.LazyFrame

        core_checks = [
            (self.check_column_presence, (check_lf, schema, column_info)),
            (self.check_column_values_are_unique, (sample_lf, schema)),
            (
                self.run_schema_component_checks,
                (sample_lf, schema, components, lazy),
            ),
            (self.run_index_checks, (check_lf, schema, lazy)),
            (self.run_checks, (sample_lf, schema)),
        ]

        # When drop_invalid_rows=True, data checks must run even for lazy/SQL
        # backends that default to SCHEMA_ONLY validation depth. Force
        # SCHEMA_AND_DATA so @validate_scope(DATA) checks are not skipped.
        _check_ctx = (
            config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA)
            if getattr(schema, "drop_invalid_rows", False)
            else config_context()
        )

        with _check_ctx:
            for check, args in core_checks:
                results = check(*args)  # type: ignore[operator]
                if isinstance(results, CoreCheckResult):
                    results = [results]

                for result in results:
                    if result.passed:
                        continue

                    if result.schema_error is not None:
                        error = result.schema_error
                    else:
                        fc = _unwrap_failure_cases(result.failure_cases)
                        error = SchemaError(
                            schema,
                            data=check_lf,
                            message=result.message,
                            failure_cases=fc,
                            check=result.check,
                            check_index=result.check_index,
                            check_output=result.check_output,
                            reason_code=result.reason_code,
                        )
                    error_handler.collect_error(
                        get_error_category(result.reason_code),
                        result.reason_code,
                        error,
                        original_exc=result.original_exc,
                    )

        if error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)
                check_obj_parsed = self.drop_invalid_rows(
                    check_obj_parsed, error_handler
                )
                return check_obj_parsed
            else:
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=_to_frame_kind_nw(check_lf, return_type),
                )

        return _to_frame_kind_nw(check_lf, return_type)

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(
        self,
        check_obj,
        schema,
    ) -> list[CoreCheckResult]:
        """Run a list of checks on the check object."""
        # dataframe-level checks
        check_results: list[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                if self.is_native_delegated_check(check) and _is_pandas_like(
                    check_obj
                ):
                    # Hypothesis / groupby dataframe-level checks on pandas
                    # frames run through the native pandas backend.
                    check_results.append(
                        self.run_native_check(
                            check_obj, schema, check, check_index
                        )
                    )
                else:
                    check_results.append(
                        self.run_check(check_obj, schema, check, check_index)
                    )
            except SchemaDefinitionError:
                raise
            except Exception as err:
                # catch other exceptions that may occur when executing the check
                err_msg = f'"{err.args[0]}"' if err.args else ""
                err_str = f"{err.__class__.__name__}({err_msg})"
                msg = (
                    f"Error while executing check function: {err_str}\n"
                    + traceback.format_exc()
                )
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=err_str,
                        original_exc=err,
                    )
                )
        return check_results

    def run_schema_component_checks(
        self,
        check_obj,
        schema,
        schema_components: list,
        lazy: bool,
    ) -> list[CoreCheckResult]:
        """Run checks for all schema components."""
        check_results = []
        # Convert to native frame for column component dispatch.
        # Column.validate() calls get_backend(check_obj) which looks up by native
        # type — native polars LazyFrame for polars schemas, ibis.Table for ibis schemas.
        native_obj = _to_native(check_obj)
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                schema_component.validate(native_obj, lazy=lazy)
                # The component validate() not raising is the success signal.
            except SchemaError as err:
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check="schema_component_checks",
                        reason_code=err.reason_code,
                        schema_error=err,
                    )
                )
            except SchemaErrors as err:
                check_results.extend(
                    [
                        CoreCheckResult(
                            passed=False,
                            check="schema_component_checks",
                            reason_code=schema_error.reason_code,
                            schema_error=schema_error,
                        )
                        for schema_error in err.schema_errors
                    ]
                )
        return check_results

    def run_native_parsers(self, check_obj, schema):
        """Run custom ``schema.parsers`` on the native pandas frame.

        Narwhals has no parser step, and custom parsers are arbitrary user code
        written against the native pandas frame. The parser function is applied
        directly here (self-contained — no dispatch to the pandas schema
        backend): dataframe-level parsers receive the whole frame, and
        ``element_wise`` parsers are applied row-wise.
        """
        for parser in schema.parsers:
            parser_fn = functools.partial(
                parser._parser_fn, **parser._parser_kwargs
            )
            if getattr(parser, "element_wise", False):
                check_obj = check_obj.apply(parser_fn, axis=1)
            else:
                check_obj = parser_fn(check_obj)
        return check_obj

    def coerce_native_index(self, check_obj, schema, error_handler):
        """Coerce the pandas index dtype via the native Index component.

        Index/MultiIndex is the one pandas concept Narwhals cannot express, so
        index coercion is delegated to the native pandas Index backend. Done on
        the native frame before wrapping so the coerced index propagates
        through Narwhals (which preserves the pandas index) to the output.
        """
        index = schema.index
        if not (
            getattr(index, "coerce", False) or getattr(schema, "coerce", False)
        ):
            return check_obj

        check_obj = check_obj.copy()
        try:
            check_obj.index = index.coerce_dtype(check_obj.index)
        except SchemaError as exc:
            error_handler.collect_error(
                get_error_category(exc.reason_code), exc.reason_code, exc
            )
        except SchemaErrors as exc:
            error_handler.collect_errors(exc.schema_errors)
        return check_obj

    def check_native_column_names_unique(self, check_obj, schema):
        """Return a ``SchemaError`` if duplicate column labels are present.

        Mirrors the native pandas ``check_column_names_are_unique``. Runs on
        the native pandas frame (before Narwhals wrapping) because Narwhals
        rejects duplicate column labels at construction time. Returns ``None``
        when the check passes or ``unique_column_names`` is not set.
        """
        if not getattr(schema, "unique_column_names", False):
            return None

        columns = getattr(check_obj, "columns", None)
        if columns is None:
            return None

        failed = columns[columns.duplicated()]
        if not failed.any():
            return None

        return SchemaError(
            schema,
            data=check_obj,
            message=(
                "dataframe contains multiple columns with label(s): "
                f"{failed.tolist()}"
            ),
            failure_cases=failed,
            check="dataframe_column_labels_unique",
            reason_code=SchemaErrorReason.DUPLICATE_COLUMN_LABELS,
        )

    def run_index_checks(
        self,
        check_obj,
        schema,
        lazy: bool,
    ) -> list[CoreCheckResult]:
        """Validate the pandas ``index`` / ``MultiIndex`` component.

        Delegates to the native pandas Index/MultiIndex backends (which stay
        registered even when the Narwhals backend is active). Narwhals
        preserves the pandas index through its operations, so ``_to_native``
        of the parsed Narwhals frame still carries the original index.

        For non-pandas frames this is a no-op — those schemas have no ``index``
        component, and the ``validate`` method already warned if one is present.
        """
        index_component = getattr(schema, "index", None)
        if index_component is None:
            return []

        native_obj = _to_native(check_obj)
        if not _native_is_pandas_like(native_obj):
            return []

        results: list[CoreCheckResult] = []
        try:
            # inplace=True: the index is read-only for validation purposes here;
            # any coercion applies to a copy the native backend manages.
            index_component.validate(native_obj, lazy=lazy, inplace=True)
        except SchemaError as err:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="index",
                    reason_code=err.reason_code,
                    schema_error=err,
                )
            )
        except SchemaErrors as err:
            results.extend(
                CoreCheckResult(
                    passed=False,
                    check="index",
                    reason_code=schema_error.reason_code,
                    schema_error=schema_error,
                )
                for schema_error in err.schema_errors
            )
        return results

    def collect_column_info(self, check_obj, schema):
        """Collect column metadata for the dataframe."""

        frame_column_names = check_obj.collect_schema().names()

        column_names: list[Any] = []
        absent_column_names: list[Any] = []
        regex_match_patterns: list[Any] = []

        for col_name, col_schema in schema.columns.items():
            if (
                not col_schema.regex
                and col_name not in frame_column_names
                and col_schema.required
            ):
                absent_column_names.append(col_name)

            if col_schema.regex:
                try:
                    column_names.extend(
                        col_schema.get_backend(
                            _to_native(check_obj)
                        ).get_regex_columns(col_schema, check_obj)
                    )
                    regex_match_patterns.append(col_schema.selector)
                except SchemaError:
                    pass
            elif col_name in frame_column_names:
                column_names.append(col_name)

        # drop adjacent duplicated column names
        destuttered_column_names = list(frame_column_names)

        return ColumnInfo(
            sorted_column_names=dict.fromkeys(column_names),
            expanded_column_names=frozenset(column_names),
            destuttered_column_names=destuttered_column_names,
            absent_column_names=absent_column_names,
            regex_match_patterns=regex_match_patterns,
        )

    def coerce_dtype(self, check_obj, schema):
        """Coerce dtypes to the schema (Narwhals-native).

        Two cases are handled:

        * **Row-wise auto_coerce dtypes** (e.g. ``PydanticModel``): coerced by
          the dtype engine itself over the whole frame — works for any backend.
        * **Column- and schema-level dtypes**: coerced with ``nw.cast`` for
          eager pandas-like frames. Narwhals normalizes pandas dtypes, so
          native pandas dtype fidelity (nullable ``Int64``, ``Categorical``,
          tz-aware datetimes) is not guaranteed. Cast failures are reported as
          ``DATATYPE_COERCION`` errors.

        Column-level coercion remains a no-op for non-pandas Narwhals backends
        (a known gap). Accepts and returns either a Narwhals frame (validate
        path) or a native frame (direct ``schema.coerce_dtype(df)`` calls).
        """
        was_wrapped = isinstance(check_obj, (nw.DataFrame, nw.LazyFrame))

        # Row-wise auto_coerce dtypes (e.g. PydanticModel): the dtype engine
        # coerces the whole frame. Self-contained (no pandas schema backend).
        if (
            schema.dtype is not None
            and schema.coerce
            and getattr(schema.dtype, "auto_coerce", False)
        ):
            config_ctx = get_config_context(validation_depth_default=None)
            coerce_fn = (
                "try_coerce"
                if config_ctx.validation_depth
                in (ValidationDepth.SCHEMA_AND_DATA, ValidationDepth.DATA_ONLY)
                else "coerce"
            )
            native_obj = _to_native(check_obj)
            try:
                coerced = getattr(schema.dtype, coerce_fn)(native_obj)
            except ParserError as exc:
                raise SchemaError(
                    schema=schema,
                    data=native_obj,
                    message=exc.args[0],
                    check=f"coerce_dtype('{schema.dtype}')",
                    reason_code=SchemaErrorReason.DATATYPE_COERCION,
                    failure_cases=exc.failure_cases,
                    check_output=exc.parser_output,
                ) from exc
            return _to_lazy_nw(coerced) if was_wrapped else coerced

        # Column-/schema-level coercion — pandas-like frames only. Hybrid:
        # plain numpy dtypes are cast with ``nw.cast`` (Narwhals-native), while
        # pandas extension dtypes (nullable Int64, Categorical, tz-aware
        # datetimes, ``string``) fall back to the pandas dtype engine so native
        # semantics are preserved. A Narwhals cast that raises also falls back.
        if not _native_is_pandas_like(_to_native(check_obj)):
            return check_obj

        lf = check_obj if was_wrapped else _to_lazy_nw(check_obj)
        frame_cols = lf.collect_schema().names()

        # Build (column, pandera_dtype) pairs to coerce.
        targets = self._collect_coerce_targets(schema, frame_cols)
        if not targets:
            return check_obj

        error_handler = ErrorHandler(lazy=True)
        # Columns that must be coerced through the pandas dtype engine (either
        # extension dtypes, or numpy dtypes whose Narwhals cast raised).
        pandas_fallback: list[tuple[Any, Any]] = []

        for col_name, col_dtype in targets:
            target_nw = (
                _narwhals_target_dtype(col_dtype)
                if self._prefers_narwhals_cast(col_dtype)
                else None
            )
            if target_nw is None:
                pandas_fallback.append((col_name, col_dtype))
                continue
            try:
                casted = lf.with_columns(nw.col(col_name).cast(target_nw))
                # Force evaluation to surface cast errors (pandas is eager, so
                # this is a cheap materialization).
                _materialize(casted.select(nw.col(col_name)))
                lf = casted
            except Exception:  # noqa: BLE001 — narwhals raises ValueError
                # Narwhals could not cast faithfully — fall back to the pandas
                # dtype engine, which produces a proper DATATYPE_COERCION error
                # (with failing values) or the correct native dtype.
                pandas_fallback.append((col_name, col_dtype))

        if pandas_fallback:
            lf = self._coerce_via_pandas_engine(
                lf, pandas_fallback, schema, error_handler
            )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=_to_native(_materialize(lf)),
            )

        return lf if was_wrapped else _to_native(_materialize(lf))

    @staticmethod
    def _prefers_narwhals_cast(pandera_dtype) -> bool:
        """True if a dtype round-trips faithfully through ``nw.cast``.

        Plain numpy dtypes (int, float, str, bool) are cast Narwhals-native.
        pandas extension dtypes (nullable ``Int64``, ``Categorical``, tz-aware
        datetimes, ``string``) are handled by the pandas dtype engine instead.
        """
        from pandera.engines import numpy_engine

        return isinstance(pandera_dtype, numpy_engine.DataType)

    def _collect_coerce_targets(self, schema, frame_cols):
        """Resolve ``(column_name, pandera_dtype)`` pairs to coerce."""
        targets: list[tuple[Any, Any]] = []
        seen: set = set()
        for col_name, col_schema in schema.columns.items():
            if not (
                getattr(col_schema, "coerce", False)
                or getattr(schema, "coerce", False)
            ):
                continue
            col_dtype = (
                schema.dtype if schema.dtype is not None else col_schema.dtype
            )
            if col_dtype is None:
                continue
            if getattr(col_schema, "regex", False):
                matches = [
                    c
                    for c in frame_cols
                    if re.search(col_schema.selector, str(c))
                ]
            elif col_name in frame_cols:
                matches = [col_name]
            else:
                matches = []
            for c in matches:
                if c not in seen:
                    targets.append((c, col_dtype))
                    seen.add(c)

        # Schema-level dtype with no explicit columns: coerce every column.
        if schema.dtype is not None and schema.coerce and not schema.columns:
            for c in frame_cols:
                if c not in seen:
                    targets.append((c, schema.dtype))
                    seen.add(c)
        return targets

    def _coerce_via_pandas_engine(self, lf, columns, schema, error_handler):
        """Coerce ``columns`` through the pandas dtype engine (native fidelity).

        Used for pandas extension dtypes and as a fallback when ``nw.cast``
        cannot coerce a column. Operates on the native pandas frame (the
        Narwhals frame is materialized first — pandas is eager, so this is
        cheap) and preserves the index. Coercion failures are collected as
        ``DATATYPE_COERCION`` errors with the offending values.
        """
        native = nw.to_native(_materialize(lf)).copy()
        for col_name, col_dtype in columns:
            # ``try_coerce`` raises ``ParserError`` (with failing values) on
            # failure, unlike ``coerce`` which surfaces a raw backend error.
            try:
                native[col_name] = col_dtype.try_coerce(native[col_name])
            except ParserError as exc:
                error_handler.collect_error(
                    get_error_category(SchemaErrorReason.DATATYPE_COERCION),
                    SchemaErrorReason.DATATYPE_COERCION,
                    SchemaError(
                        schema=schema,
                        data=native,
                        message=(
                            f"Error while coercing '{col_name}' to type "
                            f"{col_dtype}: {exc}"
                        ),
                        failure_cases=exc.failure_cases,
                        check=f"coerce_dtype('{col_dtype}')",
                        reason_code=SchemaErrorReason.DATATYPE_COERCION,
                    ),
                )
        return _to_lazy_nw(native)

    def collect_schema_components(
        self,
        check_obj,
        schema,
        column_info: ColumnInfo,
    ):
        """Collects all schema components to use for validation."""

        columns: dict = schema.columns
        frame_column_names = check_obj.collect_schema().names()

        # Row-wise dtypes (e.g. PydanticModel, auto_coerce=True) apply to the
        # whole row and are handled by coerce_dtype at the dataframe level.
        # Per-column components must not be created for them — the per-column
        # dtype check would incorrectly compare each column's native type
        # against the row-wise dtype class.
        is_row_dtype = schema.dtype is not None and getattr(
            schema.dtype, "auto_coerce", False
        )

        if (
            not schema.columns
            and schema.dtype is not None
            and not is_row_dtype
        ):
            # set schema components to dataframe dtype if columns are not
            # specified but the dataframe-level dtype is specified.
            columns = {
                col_name: col
                for col_name, col in zip(
                    frame_column_names,
                    schema.infer_columns(frame_column_names),
                )
            }

        schema_components = []
        for col_name, col in columns.items():
            if (
                col.required  # type: ignore
                or col_name in frame_column_names
                or (
                    column_info.regex_match_patterns is not None
                    and col.selector in column_info.regex_match_patterns
                )
            ) and col_name not in column_info.absent_column_names:
                col = copy.deepcopy(col)
                if schema.dtype is not None:
                    # override column dtype with dataframe dtype
                    col.dtype = schema.dtype  # type: ignore

                # Warn once per column when coerce=True was requested but will
                # not be applied — for non-pandas Narwhals backends the
                # ColumnBackend has no coerce step, so column-level coerce=True
                # is a no-op that would otherwise silently produce a
                # WRONG_DATATYPE error. Pandas frames are coerced Narwhals-native
                # in ``coerce_dtype``, so no warning is needed there.
                if getattr(col, "coerce", False) and not _is_pandas_like(
                    check_obj
                ):
                    warn_col_name = getattr(col, "name", None) or getattr(
                        col, "selector", col_name
                    )
                    warnings.warn(
                        f"coerce=True is not applied by the narwhals backend for "
                        f"column '{warn_col_name}'. The column dtype will not be "
                        f"coerced; any dtype mismatch will be reported as a "
                        f"WRONG_DATATYPE error.",
                        SchemaWarning,
                        stacklevel=8,
                    )
                # disable coercion at the schema component level since the
                # dataframe-level schema already coerced it.
                col.coerce = False  # type: ignore
                schema_components.append(col)

        return schema_components

    ###########
    # Parsers #
    ###########

    @staticmethod
    def _has_default(col_schema) -> bool:
        """True if the column schema declares a (non-null) default value."""
        default = col_schema.default
        # None or NaN (a float that is not equal to itself) means "no default".
        return default is not None and not (
            isinstance(default, float) and default != default
        )

    def add_missing_columns(self, check_obj, schema, column_info: ColumnInfo):
        """Add schema columns missing from the frame.

        Absent columns must either declare a default value or be nullable;
        otherwise an ``ADD_MISSING_COLUMN_NO_DEFAULT`` error is raised. Missing
        columns are inserted in schema order relative to the existing columns.

        Column construction is hybrid, mirroring the coerce path: plain numpy
        dtypes with a concrete default are built Narwhals-native
        (``nw.lit(value).cast(...)``); extension dtypes (nullable ``Int64`` etc.)
        and null-valued columns are given their schema dtype through the pandas
        dtype engine, since ``nw.cast`` cannot represent them (e.g. a null
        integer).
        """
        if not (
            column_info.absent_column_names
            and getattr(schema, "add_missing_columns", False)
        ):
            return check_obj

        original_cols = check_obj.collect_schema().names()

        absent_errors: list[SchemaError] = []
        add_exprs = []
        # (col_name, pandera_dtype) for columns whose dtype must be fixed
        # through the pandas dtype engine after the Narwhals literals are added.
        native_dtype_fix: list[tuple[Any, Any]] = []
        for col_name in column_info.absent_column_names:
            col_schema = schema.columns[col_name]
            has_default = self._has_default(col_schema)
            if not has_default and not col_schema.nullable:
                absent_errors.append(
                    SchemaError(
                        schema=schema,
                        data=_to_native(check_obj),
                        message=(
                            f"column '{col_name}' in "
                            f"{schema.__class__.__name__} {schema.columns} "
                            "requires a default value when non-nullable "
                            "add_missing_columns is enabled"
                        ),
                        failure_cases=col_name,
                        check="add_missing_has_default",
                        reason_code=(
                            SchemaErrorReason.ADD_MISSING_COLUMN_NO_DEFAULT
                        ),
                    )
                )
                continue
            value = col_schema.default if has_default else None
            expr = nw.lit(value)
            target_nw = (
                _narwhals_target_dtype(col_schema.dtype)
                if value is not None
                and self._prefers_narwhals_cast(col_schema.dtype)
                else None
            )
            if target_nw is not None:
                expr = expr.cast(target_nw)
            elif col_schema.dtype is not None:
                # Extension dtype or null value: add the literal now, then give
                # it the correct native dtype below.
                native_dtype_fix.append((col_name, col_schema.dtype))
            add_exprs.append(expr.alias(col_name))

        if absent_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=absent_errors,
                data=_to_native(check_obj),
            )

        if add_exprs:
            check_obj = check_obj.with_columns(add_exprs)
            ordered = self._order_columns_with_missing(
                original_cols, schema, column_info
            )
            check_obj = check_obj.select(ordered)

        if native_dtype_fix:
            native = nw.to_native(_materialize(check_obj)).copy()
            for col_name, col_dtype in native_dtype_fix:
                native[col_name] = col_dtype.try_coerce(native[col_name])
            check_obj = _to_lazy_nw(native)
        return check_obj

    @staticmethod
    def _order_columns_with_missing(original_cols, schema, column_info):
        """Compute column order after inserting missing columns.

        Ports the native pandas ordering: missing schema columns are inserted
        at their schema position relative to the existing columns, without
        disturbing the order of columns already present.
        """
        absent = column_info.absent_column_names
        schema_cols: dict[Any, None] = {}
        for col_name, col_schema in schema.columns.items():
            if col_name in original_cols or col_schema.required:
                schema_cols[col_name] = None

        ordered: list[Any] = []
        for col_name in original_cols:
            pop_cols = []
            for next_col_name in iter(schema_cols):
                if next_col_name in absent and next_col_name not in ordered:
                    ordered.append(next_col_name)
                    pop_cols.append(next_col_name)
                else:
                    for pop_col in pop_cols:
                        schema_cols.pop(pop_col)
                    break
            ordered.append(col_name)
            schema_cols.pop(col_name, None)

        for col_name in absent:
            if col_name not in ordered:
                ordered.append(col_name)
        return ordered

    def set_defaults(self, check_obj, schema):
        """Fill null values in columns that declare a default (Narwhals-native)."""
        frame_cols = check_obj.collect_schema().names()
        exprs = []
        for col_name, col_schema in schema.columns.items():
            if not self._has_default(col_schema) or col_name not in frame_cols:
                continue
            expr = nw.col(col_name).fill_null(col_schema.default)
            target = _narwhals_target_dtype(col_schema.dtype)
            if target is not None:
                expr = expr.cast(target)
            exprs.append(expr.alias(col_name))
        if exprs:
            check_obj = check_obj.with_columns(exprs)
        return check_obj

    def strict_filter_columns(
        self,
        check_obj,
        schema,
        column_info: ColumnInfo,
    ):
        """Filter columns that aren't specified in the schema."""
        # dataframe strictness check makes sure all columns in the dataframe
        # are specified in the dataframe schema
        if not (schema.strict or schema.ordered):
            return check_obj

        filter_out_columns = []
        sorted_column_names = iter(column_info.sorted_column_names)
        for column in column_info.destuttered_column_names:
            is_schema_col = column in column_info.expanded_column_names
            if schema.strict is True and not is_schema_col:
                raise SchemaError(
                    schema=schema,
                    data=check_obj,
                    message=(
                        f"column '{column}' not in {schema.__class__.__name__}"
                        f" {schema.columns}"
                    ),
                    failure_cases=column,
                    check="column_in_schema",
                    reason_code=SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
                )
            if schema.strict == "filter" and not is_schema_col:
                filter_out_columns.append(column)
            if schema.ordered and is_schema_col:
                try:
                    next_ordered_col = next(sorted_column_names)
                except StopIteration:
                    raise SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=f"column '{column}' out-of-order",
                        failure_cases=column,
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    )
                else:
                    if next_ordered_col != column:
                        raise SchemaError(
                            schema=schema,
                            data=check_obj,
                            message=f"column '{column}' out-of-order",
                            failure_cases=column,
                            check="column_ordered",
                            reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                        )

        if schema.strict == "filter":
            check_obj = check_obj.drop(filter_out_columns)

        return check_obj

    ##########
    # Checks #
    ##########

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_column_presence(
        self,
        check_obj,
        schema,
        column_info: Any,
    ) -> list[CoreCheckResult]:
        """Check that all columns in the schema are present in the dataframe."""
        results = []
        if column_info.absent_column_names and not schema.add_missing_columns:
            for colname in column_info.absent_column_names:
                if is_regex(colname):
                    # don't raise an error if the column schema name is a
                    # regex pattern — try to select using regex expression
                    try:
                        frame_cols = check_obj.collect_schema().names()
                        matching = [
                            c for c in frame_cols if re.search(colname, c)
                        ]
                        if matching:
                            continue
                    except Exception:
                        pass
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="column_in_dataframe",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        message=(
                            f"column '{colname}' not in dataframe"
                            f"\n{nw.to_native(_materialize(check_obj.head()))}"
                        ),
                        failure_cases=colname,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def check_column_values_are_unique(
        self,
        check_obj,
        schema,
    ) -> CoreCheckResult:
        """Check that column values are unique."""

        passed = True
        message = None
        failure_cases = None

        if not schema.unique:
            return CoreCheckResult(
                passed=passed,
                check="multiple_fields_uniqueness",
            )

        temp_unique: list[list] = (
            [schema.unique]
            if all(isinstance(x, str) for x in schema.unique)
            else schema.unique
        )
        frame_column_names = check_obj.collect_schema().names()
        check_output = None
        for lst in temp_unique:
            subset = [x for x in lst if x in frame_column_names]
            grouped = (
                check_obj.select(subset)
                .group_by(*[nw.col(c) for c in subset])
                .agg(nw.len().alias("_count"))
            )
            dup_rows = grouped.filter(nw.col("_count") > 1).drop("_count")
            # Bounded: dup_rows contains only rows with duplicate key values — not the full frame.
            # Materialization is required here to evaluate len() and produce failure_cases.
            native_dups = nw.to_native(_materialize(dup_rows))

            if len(native_dups) > 0:
                failure_cases = native_dups
                passed = False
                message = (
                    f"columns '{(*subset,)}' not unique:\n{failure_cases}"
                )
                break
        return CoreCheckResult(
            passed=passed,
            check="multiple_fields_uniqueness",
            reason_code=SchemaErrorReason.DUPLICATES,
            message=message,
            failure_cases=failure_cases,
            check_output=check_output,
        )
