"""Base schema backend for Narwhals."""

import warnings
from collections import defaultdict

import narwhals.stable.v1 as nw
import polars as pl

from pandera.api.narwhals.error_handler import ErrorHandler
from pandera.api.narwhals.utils import _to_native
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)


def _materialize(frame) -> nw.DataFrame:
    """Materialize a LazyFrame or SQL-lazy DataFrame to a Narwhals DataFrame.

    Delegates to NarwhalsCheckBackend._materialize — single implementation,
    no duplication. _materialize stays in checks.py per locked design decision.
    """
    return NarwhalsCheckBackend._materialize(frame)


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

        :param head: Number of rows to take from the head.
        :param tail: Number of rows to take from the tail.
        :param sample: Not supported — raises NotImplementedError.
        :param random_state: Ignored (no random sampling supported).
        :raises NotImplementedError: If sample is not None.
        """
        if sample is not None:
            raise NotImplementedError(
                "sample= is not supported in the Narwhals backend. "
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

        For Narwhals (Polars) inputs: materializes all frames to native types.
        For ibis inputs: preserves ibis.Table laziness for failure_cases and
        check_output — only evaluates the passed boolean via ibis .execute().
        """
        check_result = check(check_obj, *args)

        # Detect ibis check result: IbisCheckBackend returns ibis lazy types
        # (ibis BooleanScalar for check_passed, ibis.Table for failure_cases).
        # For ibis, evaluate passed via .execute() and preserve failure_cases as ibis.Table.
        _is_ibis_result = False
        try:
            import ibis as _ibis
            import ibis.expr.types as _ir
            if isinstance(
                check_result.check_passed,
                (_ir.BooleanScalar, _ir.BooleanColumn),
            ) or isinstance(check_result.failure_cases, _ibis.Table):
                _is_ibis_result = True
        except ImportError:
            pass

        if _is_ibis_result:
            # Ibis path: evaluate passed via ibis .execute() (returns Python bool or Series)
            passed_val = check_result.check_passed.execute()
            passed = bool(passed_val) if not hasattr(passed_val, '__iter__') else bool(passed_val.all())

            message = None
            failure_cases = None

            if not passed:
                if check_result.failure_cases is None:
                    failure_cases = passed
                    message = f"Check '{check}' failed — no failure cases captured."
                else:
                    # Preserve ibis.Table as-is — tests/ibis/ call .execute()/.to_pandas() on it
                    failure_cases = check_result.failure_cases
                    message = f"Check '{check}' failed."

                if check.raise_warning:
                    warnings.warn(message, SchemaWarning)
                    return CoreCheckResult(
                        passed=True,
                        check=check,
                        reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                    )

            # check_output: also ibis.Table — return as-is (ibis backend consumers expect it)
            check_output = check_result.check_output
            return CoreCheckResult(
                passed=passed,
                check=check,
                check_index=check_index,
                check_output=check_output,
                reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                message=message,
                failure_cases=failure_cases,
            )

        # Narwhals (Polars) path — materialize Narwhals frames to native types
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

        Ported from PolarsSchemaBackend.failure_cases_metadata(). By the time
        failure_cases reach this method they are already native (pl.DataFrame
        or scalar) — _to_native() was called at construction sites.

        Note: pl.LazyFrame failure_cases are not supported (Phase 4 Polars-only
        limitation).
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

            # Detect ibis-originated failure cases.
            # After Plan 05-04, ibis check results may carry failure_cases as:
            #   - ibis.Table (lazy ibis expression — from the ibis path in run_check)
            #   - pyarrow.lib.Table (materialized — from Narwhals collect() on ibis-backed frame)
            # Both must be handled before the scalar else branch to avoid Object-dtype in pl.concat.
            _ibis_fc = None
            try:
                import ibis as _ibis_mod
                if isinstance(err.failure_cases, _ibis_mod.Table):
                    _ibis_fc = err.failure_cases
            except ImportError:
                pass
            if _ibis_fc is None:
                try:
                    import pyarrow as _pa
                    if isinstance(err.failure_cases, _pa.Table):
                        # pyarrow.Table is also ibis-originated; tag for ibis branch below
                        _ibis_fc = err.failure_cases
                except ImportError:
                    pass

            if isinstance(err.failure_cases, pl.LazyFrame):
                raise NotImplementedError

            if isinstance(err.failure_cases, pl.DataFrame):
                failure_cases_df = err.failure_cases

                # get row number of the failure cases
                if err.check_output is not None:
                    if hasattr(err.check_output, "with_row_index"):
                        _index_lf = err.check_output.with_row_index("index")
                    else:
                        _index_lf = err.check_output.with_row_count("index")
                    index = _index_lf.filter(
                        pl.col(CHECK_OUTPUT_KEY).eq(False)
                    )["index"]
                else:
                    index = pl.Series("index", [None] * len(failure_cases_df), dtype=pl.Int32)

                if len(err.failure_cases.columns) > 1:
                    # for boolean dataframe check results, reduce failure cases
                    # to a struct column
                    failure_cases_df = err.failure_cases.with_columns(
                        failure_case=pl.Series(
                            err.failure_cases.rows(named=True)
                        )
                    ).select(pl.col.failure_case.struct.json_encode())
                else:
                    failure_cases_df = err.failure_cases.rename(
                        {err.failure_cases.columns[0]: "failure_case"}
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

            elif _ibis_fc is not None:
                # Ibis path: materialize ibis.Table or pyarrow.Table to pandas,
                # then convert to pl.DataFrame following the pl.DataFrame branch structure.
                if hasattr(_ibis_fc, "execute"):
                    # ibis.Table — execute() materializes to pandas
                    pd_fc = _ibis_fc.execute()
                else:
                    # pyarrow.Table — to_pandas() materializes directly
                    pd_fc = _ibis_fc.to_pandas()
                pl_fc = pl.from_pandas(pd_fc)

                # Compute index from check_output (may also be an ibis.Table)
                if err.check_output is not None:
                    try:
                        import ibis as _ibis_co
                        if isinstance(err.check_output, _ibis_co.Table):
                            check_output_pd = err.check_output.execute()
                            index = pl.Series(
                                "index",
                                [
                                    i
                                    for i, v in enumerate(
                                        check_output_pd[CHECK_OUTPUT_KEY]
                                    )
                                    if not v
                                ],
                                dtype=pl.Int32,
                            )
                        else:
                            index = pl.Series(
                                "index", [None] * len(pl_fc), dtype=pl.Int32
                            )
                    except Exception:
                        index = pl.Series(
                            "index", [None] * len(pl_fc), dtype=pl.Int32
                        )
                else:
                    index = pl.Series(
                        "index", [None] * len(pl_fc), dtype=pl.Int32
                    )

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

            else:
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

        failure_cases = pl.concat(failure_case_collection)

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
        """Remove invalid rows according to failures in error_handler.

        For Ibis: delegates to IbisSchemaBackend.drop_invalid_rows() since
        Narwhals has no positional-join / row_number abstraction for ibis.
        For Polars: uses nw.all_horizontal() to combine boolean check_outputs.

        :param check_obj: The frame to filter.
        :param error_handler: ErrorHandler whose schema_errors carry check_output.
        :returns: Filtered frame with only rows where all checks passed.
        """
        errors = getattr(error_handler, "schema_errors", [])
        if not errors:
            return check_obj

        # Detect ibis path: unwrap to native and check type
        native = nw.to_native(check_obj) if isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)) else check_obj
        try:
            import ibis as _ibis
            if isinstance(native, _ibis.Table):
                from pandera.backends.ibis.base import IbisSchemaBackend
                result = IbisSchemaBackend().drop_invalid_rows(native, error_handler)
                return nw.from_native(result, eager_or_interchange_only=False)
        except ImportError:
            pass

        # Polars path: use nw.all_horizontal() for boolean reduction (replaces pl.fold)
        check_outputs = [
            err.check_output for err in errors
            if err.check_output is not None
        ]
        if not check_outputs:
            return check_obj

        # check_outputs are native pl.DataFrame with CHECK_OUTPUT_KEY boolean column
        merged_pl = pl.DataFrame(
            {str(i): co[CHECK_OUTPUT_KEY] for i, co in enumerate(check_outputs)}
        )
        merged_nw = nw.from_native(merged_pl)
        valid_rows_nw = merged_nw.select(
            nw.all_horizontal(*[nw.col(c) for c in merged_pl.columns]).alias("valid_rows")
        )
        valid_rows = nw.to_native(valid_rows_nw)["valid_rows"]
        return check_obj.filter(valid_rows)
