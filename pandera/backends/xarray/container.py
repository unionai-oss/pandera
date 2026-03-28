"""Xarray DataArray and Dataset schema backends.

**Chunked (Dask) data**

Data-level checks run only when :attr:`~pandera.config.ValidationDepth` includes
data (``SCHEMA_AND_DATA`` or ``DATA_ONLY``). For chunked xarray objects, the API
layer defaults to ``SCHEMA_ONLY`` unless the user sets ``validation_depth`` via
``PANDERA_VALIDATION_DEPTH`` or :func:`~pandera.config.config_context` (same
pattern as Polars LazyFrame). Structural checks always run.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from pandera.api.base.error_handler import ErrorHandler, get_error_category
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.backends.base import CoreCheckResult
from pandera.backends.xarray.base import XarraySchemaBackend
from pandera.config import ValidationDepth, get_config_context
from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors


def _should_run_xarray_data_checks() -> bool:
    """Whether to run data-level :class:`~pandera.api.checks.Check` instances."""
    cfg = get_config_context()
    return cfg.validation_depth in (
        ValidationDepth.SCHEMA_AND_DATA,
        ValidationDepth.DATA_ONLY,
    )


def _collect(
    error_handler: ErrorHandler,
    schema: Any,
    check_obj: Any,
    results: list[CoreCheckResult],
) -> None:
    for result in results:
        if result.passed:
            continue
        err = result.schema_error or SchemaError(
            schema,
            data=check_obj,
            message=result.message or "",
            failure_cases=result.failure_cases,
            check=result.check,
            check_index=result.check_index,
            check_output=result.check_output,
            reason_code=result.reason_code,
        )
        assert result.reason_code is not None
        error_handler.collect_error(
            get_error_category(result.reason_code),
            result.reason_code,
            err,
            original_exc=result.original_exc,
        )


def _aligned_dims_sizes(da1: Any, da2: Any) -> bool:
    return da1.dims == da2.dims and da1.shape == da2.shape


def _broadcast_compatible(da1: Any, da2: Any) -> bool:
    """True if xarray can broadcast da1 against da2 (shared dims same size)."""
    sizes1 = dict(zip(da1.dims, da1.shape))
    sizes2 = dict(zip(da2.dims, da2.shape))
    for d in set(sizes1) & set(sizes2):
        if sizes1[d] != sizes2[d] and sizes1[d] != 1 and sizes2[d] != 1:
            return False
    return True


class DataArraySchemaBackend(XarraySchemaBackend):
    """Validate :class:`~xarray.DataArray` against :class:`DataArraySchema`."""

    def preprocess(self, check_obj, inplace: bool = False):
        # Shallow copy so coercion or parsers never mutate the caller's object.
        return check_obj if inplace else copy.copy(check_obj)

    def _structural_core(self, schema: DataArraySchema, check_obj: Any):
        import xarray as xr

        results: list[CoreCheckResult] = []

        if schema.name is not None and check_obj.name != schema.name:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="name",
                    reason_code=SchemaErrorReason.WRONG_FIELD_NAME,
                    message=(
                        f"expected name {schema.name!r}, got {check_obj.name!r}"
                    ),
                    failure_cases=check_obj.name,
                )
            )

        if schema.dims is not None:
            exp = schema.dims
            got = check_obj.dims
            if len(exp) != len(got):
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dims",
                        reason_code=SchemaErrorReason.MISMATCH_INDEX,
                        message=(
                            f"expected ndim/dims length {len(exp)} {exp!r}, "
                            f"got {len(got)} {got!r}"
                        ),
                        failure_cases=str(got),
                    )
                )
            else:
                for i, (e, g) in enumerate(zip(exp, got)):
                    if e is not None and e != g:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="dims",
                                reason_code=SchemaErrorReason.MISMATCH_INDEX,
                                message=(
                                    f"dim position {i}: expected {e!r}, "
                                    f"got {g!r}"
                                ),
                                failure_cases=str(got),
                            )
                        )
                        break

        if schema.sizes:
            for d, sz in schema.sizes.items():
                if sz is None:
                    continue
                if d not in check_obj.sizes or check_obj.sizes[d] != sz:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="sizes",
                            reason_code=SchemaErrorReason.MISMATCH_INDEX,
                            message=(
                                f"expected size {d}={sz}, "
                                f"got {check_obj.sizes.get(d)}"
                            ),
                            failure_cases=str(check_obj.sizes),
                        )
                    )

        if schema.shape is not None:
            for i, sh in enumerate(schema.shape):
                if sh is None:
                    continue
                if i >= len(check_obj.shape) or check_obj.shape[i] != sh:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="shape",
                            reason_code=SchemaErrorReason.MISMATCH_INDEX,
                            message=(
                                f"expected shape[{i}]={sh}, "
                                f"got shape {check_obj.shape}"
                            ),
                            failure_cases=str(check_obj.shape),
                        )
                    )

        if schema.dtype is not None:
            from pandera.engines import xarray_engine

            pdt = xarray_engine.Engine.dtype(schema.dtype)
            if not pdt.check(pdt, check_obj):
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check=f"dtype({schema.dtype})",
                        reason_code=SchemaErrorReason.WRONG_DATATYPE,
                        message=(
                            f"expected dtype {schema.dtype}, "
                            f"got {check_obj.dtype}"
                        ),
                        failure_cases=str(check_obj.dtype),
                    )
                )

        if schema.chunked is True and check_obj.chunks is None:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="chunked",
                    reason_code=SchemaErrorReason.INVALID_TYPE,
                    message="expected chunked (Dask) DataArray",
                    failure_cases="eager",
                )
            )
        elif schema.chunked is False and check_obj.chunks is not None:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="chunked",
                    reason_code=SchemaErrorReason.INVALID_TYPE,
                    message="expected eager DataArray, got chunked",
                    failure_cases="chunked",
                )
            )

        if schema.array_type is not None and not isinstance(
            check_obj.data,
            schema.array_type,
        ):
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="array_type",
                    reason_code=SchemaErrorReason.INVALID_TYPE,
                    message=(
                        f"expected array type {schema.array_type}, "
                        f"got {type(check_obj.data)}"
                    ),
                    failure_cases=type(check_obj.data).__name__,
                )
            )

        if not schema.nullable and check_obj.isnull().any():
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="nullable",
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                    message="non-nullable DataArray contains null values",
                    failure_cases="null",
                )
            )

        if schema.attrs:
            for ak, av in schema.attrs.items():
                if ak not in check_obj.attrs or check_obj.attrs[ak] != av:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="attrs",
                            reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            message=(
                                f"attribute mismatch {ak!r}: "
                                f"expected {av!r}, got "
                                f"{check_obj.attrs.get(ak)!r}"
                            ),
                            failure_cases=str(check_obj.attrs.get(ak)),
                        )
                    )

        if schema.strict_attrs and schema.attrs is not None:
            allowed = set(schema.attrs.keys())
            for k in check_obj.attrs:
                if k not in allowed:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="strict_attrs",
                            reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            message=f"unexpected attribute {k!r}",
                            failure_cases=k,
                        )
                    )

        expected_coord_keys: set[str] | None = None
        if schema.coords is not None:
            if isinstance(schema.coords, list):
                expected_coord_keys = set(schema.coords)
                for cn in schema.coords:
                    if cn not in check_obj.coords:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                                message=f"missing coordinate {cn!r}",
                                failure_cases=cn,
                            )
                        )
            else:
                expected_coord_keys = set(schema.coords.keys())
                for cn, cspec in schema.coords.items():
                    if cn not in check_obj.coords:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                                message=f"missing coordinate {cn!r}",
                                failure_cases=cn,
                            )
                        )
                    else:
                        results.extend(
                            self._validate_coord_on_parent(
                                check_obj,
                                cn,
                                cspec,
                                schema.strict_coords,
                            )
                        )

        if schema.strict_coords and expected_coord_keys is not None:
            for ck in check_obj.coords:
                if ck not in expected_coord_keys:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="strict_coords",
                            reason_code=SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
                            message=f"unexpected coordinate {ck!r}",
                            failure_cases=ck,
                        )
                    )

        return results

    def _validate_coord_on_parent(
        self,
        parent: Any,
        coord_name: str,
        spec: Any,
        _parent_strict_coords: bool | str,
    ) -> list[CoreCheckResult]:
        results: list[CoreCheckResult] = []
        coord_da = parent.coords[coord_name]
        parent_dims = set(parent.dims)

        if isinstance(spec, Coordinate):
            c = spec
            if c.dimension is True and coord_name not in parent_dims:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dimension_coord",
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        message=(
                            f"coordinate {coord_name!r} must be a dimension "
                            f"coordinate on parent"
                        ),
                        failure_cases=coord_name,
                    )
                )
            if c.dimension is False and coord_name in parent_dims:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="aux_coord",
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        message=(
                            f"coordinate {coord_name!r} must not be a "
                            f"dimension of the parent"
                        ),
                        failure_cases=coord_name,
                    )
                )
            if c.indexed is True and coord_name not in parent.xindexes:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="indexed",
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        message=(
                            f"coordinate {coord_name!r} expected indexed on "
                            f"parent"
                        ),
                        failure_cases=coord_name,
                    )
                )
            if c.indexed is False and coord_name in parent.xindexes:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="indexed",
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                        message=(
                            f"coordinate {coord_name!r} expected non-indexed "
                            f"on parent"
                        ),
                        failure_cases=coord_name,
                    )
                )
            sub = c.to_data_array_schema(coord_name)
            try:
                self.validate(
                    coord_da,
                    sub,
                    lazy=True,
                    inplace=True,
                    head=None,
                    tail=None,
                    sample=None,
                    random_state=None,
                )
            except SchemaErrors as exc:
                for e in exc.schema_errors:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="coord_schema",
                            reason_code=e.reason_code,
                            schema_error=e,
                        )
                    )
            except SchemaError as e:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="coord_schema",
                        reason_code=e.reason_code,
                        schema_error=e,
                    )
                )
        else:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="coord_spec",
                    reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                    message=f"invalid coordinate spec type: {type(spec)!r}",
                    failure_cases=str(spec),
                )
            )

        return results

    def validate(
        self,
        check_obj,
        schema: DataArraySchema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        error_handler = ErrorHandler(lazy)
        check_obj = self.preprocess(check_obj, inplace=inplace)

        try:
            check_obj = self.run_parsers(schema, check_obj)
        except SchemaError as exc:
            error_handler.collect_error(
                get_error_category(exc.reason_code),
                exc.reason_code,
                exc,
            )

        if schema.coerce and schema.dtype is not None:
            try:
                check_obj = self.coerce_dtype(check_obj, schema)
            except SchemaError as exc:
                error_handler.collect_error(
                    get_error_category(exc.reason_code),
                    exc.reason_code,
                    exc,
                )

        samp = self.subsample(
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
        )

        _collect(
            error_handler,
            schema,
            check_obj,
            self._structural_core(schema, check_obj),
        )

        if _should_run_xarray_data_checks():
            _collect(
                error_handler,
                schema,
                check_obj,
                self.run_checks(samp, schema),
            )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )
        return check_obj


class DatasetSchemaBackend(XarraySchemaBackend):
    """Validate :class:`~xarray.Dataset` against :class:`DatasetSchema`."""

    def preprocess(self, check_obj, inplace: bool = False):
        # Shallow copy so validation never mutates the caller's Dataset.
        return check_obj if inplace else copy.copy(check_obj)

    def _dataset_level_structural(
        self,
        schema: DatasetSchema,
        ds: Any,
    ) -> list[CoreCheckResult]:
        results: list[CoreCheckResult] = []

        if schema.dims is not None:
            exp = set(schema.dims)
            got = set(ds.dims)
            if exp != got:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dims",
                        reason_code=SchemaErrorReason.MISMATCH_INDEX,
                        message=f"expected dims {exp!r}, got {got!r}",
                        failure_cases=str(got),
                    )
                )

        if schema.sizes:
            for d, sz in schema.sizes.items():
                if sz is None:
                    continue
                if d not in ds.sizes or ds.sizes[d] != sz:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="sizes",
                            reason_code=SchemaErrorReason.MISMATCH_INDEX,
                            message=(
                                f"dataset size {d!r} expected {sz}, "
                                f"got {ds.sizes.get(d)}"
                            ),
                            failure_cases=str(ds.sizes),
                        )
                    )

        if schema.attrs:
            for ak, av in schema.attrs.items():
                if ak not in ds.attrs or ds.attrs[ak] != av:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="attrs",
                            reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            message=(
                                f"dataset attribute {ak!r}: "
                                f"expected {av!r}, got {ds.attrs.get(ak)!r}"
                            ),
                            failure_cases=str(ds.attrs.get(ak)),
                        )
                    )

        if schema.strict_attrs and schema.attrs is not None:
            allowed = set(schema.attrs.keys())
            for k in ds.attrs:
                if k not in allowed:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="strict_attrs",
                            reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                            message=f"unexpected attribute {k!r}",
                            failure_cases=k,
                        )
                    )

        if schema.coords is not None:
            if isinstance(schema.coords, list):
                for cn in schema.coords:
                    if cn not in ds.coords:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                                message=f"missing coordinate {cn!r}",
                                failure_cases=cn,
                            )
                        )
            else:
                backend_da = DataArraySchemaBackend()
                for cn, cspec in schema.coords.items():
                    if cn not in ds.coords:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                                message=f"missing coordinate {cn!r}",
                                failure_cases=cn,
                            )
                        )
                    elif isinstance(cspec, Coordinate):
                        results.extend(
                            backend_da._validate_coord_on_parent(
                                ds,
                                cn,
                                cspec,
                                schema.strict_coords,
                            )
                        )
                    else:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                                message=f"invalid coord spec: {type(cspec)!r}",
                                failure_cases=str(cspec),
                            )
                        )

        if schema.coords is not None and schema.strict_coords:
            if isinstance(schema.coords, dict):
                allowed_c = set(schema.coords.keys())
            else:
                allowed_c = set(schema.coords)
            for ck in ds.coords:
                if ck not in allowed_c:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="strict_coords",
                            reason_code=SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
                            message=f"unexpected coordinate {ck!r}",
                            failure_cases=ck,
                        )
                    )

        return results

    def _resolve_var_name(
        self, logical: str, spec: DataVar | DataArraySchema | None
    ) -> str:
        if spec is None or not isinstance(spec, DataVar):
            return logical
        return spec.alias or logical

    def _apply_default(
        self,
        ds: Any,
        actual_name: str,
        spec: DataVar,
        dataset_schema: DatasetSchema,
    ) -> Any:
        import xarray as xr

        if spec.default is None:
            return ds
        if isinstance(spec.default, xr.DataArray):
            return ds.assign(**{actual_name: spec.default})
        if spec.dims is None:
            raise SchemaError(
                dataset_schema,
                data=ds,
                message=(
                    "inserting scalar default requires DataVar.dims "
                    f"for {actual_name!r}"
                ),
                reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
            )
        shape = []
        coords = {}
        for d in spec.dims:
            if d is None:
                continue
            if d not in ds.sizes:
                raise SchemaError(
                    dataset_schema,
                    data=ds,
                    message=f"cannot fill default: dim {d!r} not on dataset",
                    reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                )
            shape.append(ds.sizes[d])
            if d in ds.coords:
                coords[d] = ds.coords[d]
        arr = np.full(shape, spec.default)
        da = xr.DataArray(
            arr, dims=[d for d in spec.dims if d is not None], coords=coords
        )
        return ds.assign(**{actual_name: da})

    def validate(
        self,
        check_obj,
        schema: DatasetSchema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        import xarray as xr

        error_handler = ErrorHandler(lazy)
        ds = self.preprocess(check_obj, inplace=inplace)
        da_backend = DataArraySchemaBackend()

        logical_to_actual: dict[str, str] = {}
        for logical, spec in schema.data_vars.items():
            actual = self._resolve_var_name(logical, spec)
            logical_to_actual[logical] = actual

        actual_to_logicals: dict[str, list[str]] = {}
        for logical, actual in logical_to_actual.items():
            actual_to_logicals.setdefault(actual, []).append(logical)
        dupes = {
            a: lgs
            for a, lgs in actual_to_logicals.items()
            if len(lgs) > 1
        }
        if dupes:
            detail = ", ".join(
                f"{a!r} <- {lgs}" for a, lgs in dupes.items()
            )
            raise SchemaError(
                schema,
                data=ds,
                message=(
                    "multiple data_vars resolve to the same "
                    f"actual variable name: {detail}"
                ),
                reason_code=SchemaErrorReason.DUPLICATES,
            )

        planned = {logical_to_actual[k] for k in schema.data_vars}
        extras = [v for v in ds.data_vars if v not in planned]

        if schema.strict is True and extras:
            err = SchemaError(
                schema,
                data=ds,
                message=f"unexpected data variables: {extras}",
                failure_cases=str(extras),
                reason_code=SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
            )
            error_handler.collect_error(
                get_error_category(err.reason_code),
                err.reason_code,
                err,
            )
        elif schema.strict == "filter" and extras:
            ds = ds.drop_vars(extras)

        try:
            ds = self.run_parsers(schema, ds)
        except SchemaError as exc:
            error_handler.collect_error(
                get_error_category(exc.reason_code),
                exc.reason_code,
                exc,
            )

        collect_fn: Callable[..., None] = lambda res: _collect(
            error_handler, schema, ds, res
        )
        collect_fn(self._dataset_level_structural(schema, ds))

        for logical, spec in schema.data_vars.items():
            actual = logical_to_actual[logical]
            if actual not in ds.data_vars:
                if spec is None:
                    err = SchemaError(
                        schema,
                        data=ds,
                        message=f"missing required data_var {actual!r}",
                        failure_cases=actual,
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                    )
                    error_handler.collect_error(
                        get_error_category(err.reason_code),
                        err.reason_code,
                        err,
                    )
                    continue
                if isinstance(spec, DataVar):
                    if spec.required:
                        err = SchemaError(
                            schema,
                            data=ds,
                            message=f"missing required data_var {actual!r}",
                            failure_cases=actual,
                            reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        )
                        error_handler.collect_error(
                            get_error_category(err.reason_code),
                            err.reason_code,
                            err,
                        )
                        continue
                    if spec.default is not None:
                        try:
                            ds = self._apply_default(ds, actual, spec, schema)
                        except SchemaError as exc:
                            error_handler.collect_error(
                                get_error_category(exc.reason_code),
                                exc.reason_code,
                                exc,
                            )
                    if actual not in ds.data_vars:
                        continue
                else:
                    err = SchemaError(
                        schema,
                        data=ds,
                        message=f"missing required data_var {actual!r}",
                        failure_cases=actual,
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                    )
                    error_handler.collect_error(
                        get_error_category(err.reason_code),
                        err.reason_code,
                        err,
                    )
                    continue

            if spec is None:
                continue

            # Copy before set_name so shared DataVar / schema objects in the
            # user's DatasetSchema are not mutated across validate() calls.
            if isinstance(spec, DataArraySchema):
                sub_schema = copy.copy(spec)
            else:
                sub_schema = copy.copy(spec.to_data_array_schema(logical))
            sub_schema = sub_schema.set_name(actual)

            var_obj = ds[actual]
            try:
                da_backend.validate(
                    var_obj,
                    sub_schema,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except SchemaErrors as exc:
                for e in exc.schema_errors:
                    collect_fn(
                        [
                            CoreCheckResult(
                                passed=False,
                                check=actual,
                                reason_code=e.reason_code,
                                schema_error=e,
                            )
                        ]
                    )
            except SchemaError as e:
                collect_fn(
                    [
                        CoreCheckResult(
                            passed=False,
                            check=actual,
                            reason_code=e.reason_code,
                            schema_error=e,
                        )
                    ]
                )

        for logical, spec in schema.data_vars.items():
            if not isinstance(spec, DataVar):
                continue
            actual = logical_to_actual[logical]
            if actual not in ds.data_vars:
                continue
            da_self = ds[actual]
            peers_a = spec.aligned_with or ()
            peers_b = spec.broadcastable_with or ()
            for other_log in peers_a:
                other = logical_to_actual.get(other_log, other_log)
                if other not in ds.data_vars:
                    err = SchemaError(
                        schema,
                        data=ds,
                        message=(
                            f"aligned_with: peer {other_log!r} "
                            f"({other!r}) missing"
                        ),
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                    )
                    error_handler.collect_error(
                        get_error_category(err.reason_code),
                        err.reason_code,
                        err,
                    )
                    continue
                if not _aligned_dims_sizes(da_self, ds[other]):
                    err = SchemaError(
                        schema,
                        data=ds,
                        message=(
                            f"{actual!r} not aligned with {other!r} "
                            f"(dims/sizes)"
                        ),
                        reason_code=SchemaErrorReason.MISMATCH_INDEX,
                    )
                    error_handler.collect_error(
                        get_error_category(err.reason_code),
                        err.reason_code,
                        err,
                    )
            for other_log in peers_b:
                other = logical_to_actual.get(other_log, other_log)
                if other not in ds.data_vars:
                    err = SchemaError(
                        schema,
                        data=ds,
                        message=(
                            f"broadcastable_with: peer {other_log!r} missing"
                        ),
                        reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                    )
                    error_handler.collect_error(
                        get_error_category(err.reason_code),
                        err.reason_code,
                        err,
                    )
                    continue
                if not _broadcast_compatible(da_self, ds[other]):
                    err = SchemaError(
                        schema,
                        data=ds,
                        message=(
                            f"{actual!r} not broadcast-compatible with "
                            f"{other!r}"
                        ),
                        reason_code=SchemaErrorReason.MISMATCH_INDEX,
                    )
                    error_handler.collect_error(
                        get_error_category(err.reason_code),
                        err.reason_code,
                        err,
                    )

        if _should_run_xarray_data_checks():
            _collect(error_handler, schema, ds, self.run_checks(ds, schema))

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=ds,
            )
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"expected Dataset, got {type(ds)}")
        return ds
