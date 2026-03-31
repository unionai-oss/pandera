"""Xarray DataArray and Dataset schema backends.

**Chunked (Dask) data**

Data-level checks run only when :attr:`~pandera.config.ValidationDepth` includes
data (``SCHEMA_AND_DATA`` or ``DATA_ONLY``). For chunked xarray objects, the API
layer defaults to ``SCHEMA_ONLY`` unless the user sets ``validation_depth`` via
``PANDERA_VALIDATION_DEPTH`` or :func:`~pandera.config.config_context` (same
pattern as Polars LazyFrame).
"""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

from pandera.api.base.error_handler import ErrorHandler, get_error_category
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.backends.base import CoreCheckResult
from pandera.backends.xarray.base import XarraySchemaBackend
from pandera.config import ValidationScope
from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors
from pandera.validation_depth import validate_scope


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
    """True if xarray can broadcast *da1* against *da2*."""
    sizes1 = dict(zip(da1.dims, da1.shape))
    sizes2 = dict(zip(da2.dims, da2.shape))
    for d in set(sizes1) & set(sizes2):
        if sizes1[d] != sizes2[d] and sizes1[d] != 1 and sizes2[d] != 1:
            return False
    return True


def _is_pydantic_model_class(obj: Any) -> bool:
    """True if *obj* is a pydantic BaseModel **class** (not an instance)."""
    try:
        from pydantic import BaseModel

        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except ImportError:
        return False


def _validate_attrs_with_pydantic(
    attrs_model: type,
    attrs_dict: dict[str, Any],
    *,
    prefix: str = "",
) -> list[CoreCheckResult]:
    """Validate *attrs_dict* against a pydantic BaseModel class.

    Returns one :class:`CoreCheckResult` per pydantic validation error,
    with error messages formatted consistently with pandera's attr
    reporting style.
    """
    from pydantic import ValidationError

    results: list[CoreCheckResult] = []
    try:
        attrs_model(**attrs_dict)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            attr_label = f"{prefix}{loc}" if loc else prefix or "attrs"
            results.append(
                CoreCheckResult(
                    passed=False,
                    check="attrs",
                    reason_code=SchemaErrorReason.SCHEMA_COMPONENT_CHECK,
                    message=(
                        f"{attr_label}: {err['msg']} "
                        f"[type={err['type']}]"
                    ),
                    failure_cases=str(
                        _nested_get(attrs_dict, err["loc"])
                    ),
                )
            )
    return results


def _nested_get(d: Any, loc: tuple) -> Any:
    """Traverse *d* following pydantic error ``loc`` path."""
    for key in loc:
        try:
            d = d[key]
        except (KeyError, IndexError, TypeError):
            return "<missing>"
    return d


def _match_attr_value(expected: Any, actual: Any) -> bool:
    """Match an attribute value using equality, regex, or callable."""
    if callable(expected) and not isinstance(expected, type):
        return bool(expected(actual))
    if isinstance(expected, str) and expected.startswith("^"):
        return re.fullmatch(expected, str(actual)) is not None
    return actual == expected


def _run_core_checks(
    error_handler: ErrorHandler,
    schema: Any,
    check_obj: Any,
    core_checks: list[tuple],
) -> None:
    """Execute *core_checks* and feed failures into *error_handler*.

    Each entry is ``(check_fn, args)`` where *check_fn* returns
    :class:`CoreCheckResult` or ``list[CoreCheckResult]``.
    """
    for check_fn, args in core_checks:
        results = check_fn(*args)
        if isinstance(results, CoreCheckResult):
            results = [results]
        _collect(error_handler, schema, check_obj, results)


# -------------------------------------------------------------------
# DataArray backend
# -------------------------------------------------------------------


class DataArraySchemaBackend(XarraySchemaBackend):
    """Validate :class:`~xarray.DataArray` against
    :class:`DataArraySchema`."""

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj if inplace else copy.copy(check_obj)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_name(
        self, check_obj, schema: DataArraySchema
    ) -> CoreCheckResult:
        """Check that the DataArray name matches the schema."""
        if schema.name is None or check_obj.name == schema.name:
            return CoreCheckResult(passed=True, check="name")
        return CoreCheckResult(
            passed=False,
            check="name",
            reason_code=SchemaErrorReason.WRONG_FIELD_NAME,
            message=(f"expected name {schema.name!r}, got {check_obj.name!r}"),
            failure_cases=check_obj.name,
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dims(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check dimension names (and order when ordered)."""
        results: list[CoreCheckResult] = []
        if schema.dims is None:
            return results
        exp = schema.dims
        got = check_obj.dims
        ordered = getattr(schema, "ordered_dims", True)
        if ordered:
            if len(exp) != len(got):
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dims",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"expected ndim/dims length "
                            f"{len(exp)} {exp!r}, "
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
                                reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                                message=(
                                    f"dim position {i}: "
                                    f"expected {e!r}, "
                                    f"got {g!r}"
                                ),
                                failure_cases=str(got),
                            )
                        )
                        break
        else:
            exp_names = {e for e in exp if e is not None}
            got_names = set(got)
            missing = exp_names - got_names
            if missing:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dims",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"missing dims "
                            f"{sorted(missing)!r} "
                            f"(expected {exp!r}, "
                            f"got {got!r})"
                        ),
                        failure_cases=str(got),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_sizes(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check dimension sizes."""
        results: list[CoreCheckResult] = []
        if not schema.sizes:
            return results
        for d, sz in schema.sizes.items():
            if sz is None:
                continue
            if d not in check_obj.sizes or check_obj.sizes[d] != sz:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="sizes",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"expected size {d}={sz}, "
                            f"got {check_obj.sizes.get(d)}"
                        ),
                        failure_cases=str(check_obj.sizes),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_shape(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check positional shape."""
        results: list[CoreCheckResult] = []
        if schema.shape is None:
            return results
        for i, sh in enumerate(schema.shape):
            if sh is None:
                continue
            if i >= len(check_obj.shape) or check_obj.shape[i] != sh:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="shape",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"expected shape[{i}]={sh}, "
                            f"got shape "
                            f"{check_obj.shape}"
                        ),
                        failure_cases=str(check_obj.shape),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(
        self, check_obj, schema: DataArraySchema
    ) -> CoreCheckResult:
        """Check the data type."""
        if schema.dtype is None:
            return CoreCheckResult(passed=True, check="dtype")
        from pandera.engines import xarray_engine

        pdt = xarray_engine.Engine.dtype(schema.dtype)
        if pdt.check(pdt, check_obj):
            return CoreCheckResult(
                passed=True,
                check=f"dtype({schema.dtype})",
            )
        return CoreCheckResult(
            passed=False,
            check=f"dtype({schema.dtype})",
            reason_code=SchemaErrorReason.WRONG_DATATYPE,
            message=(f"expected dtype {schema.dtype}, got {check_obj.dtype}"),
            failure_cases=str(check_obj.dtype),
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_chunked(
        self, check_obj, schema: DataArraySchema
    ) -> CoreCheckResult:
        """Check chunked / eager expectation."""
        if schema.chunked is True and check_obj.chunks is None:
            return CoreCheckResult(
                passed=False,
                check="chunked",
                reason_code=SchemaErrorReason.INVALID_TYPE,
                message="expected chunked (Dask) DataArray",
                failure_cases="eager",
            )
        if schema.chunked is False and check_obj.chunks is not None:
            return CoreCheckResult(
                passed=False,
                check="chunked",
                reason_code=SchemaErrorReason.INVALID_TYPE,
                message=("expected eager DataArray, got chunked"),
                failure_cases="chunked",
            )
        return CoreCheckResult(passed=True, check="chunked")

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_array_type(
        self, check_obj, schema: DataArraySchema
    ) -> CoreCheckResult:
        """Check the underlying array type."""
        if schema.array_type is None or isinstance(
            check_obj.data, schema.array_type
        ):
            return CoreCheckResult(passed=True, check="array_type")
        return CoreCheckResult(
            passed=False,
            check="array_type",
            reason_code=SchemaErrorReason.INVALID_TYPE,
            message=(
                f"expected array type "
                f"{schema.array_type}, "
                f"got {type(check_obj.data)}"
            ),
            failure_cases=type(check_obj.data).__name__,
        )

    @validate_scope(scope=ValidationScope.DATA)
    def check_nullable(
        self, check_obj, schema: DataArraySchema
    ) -> CoreCheckResult:
        """Check for null values when nullable is False."""
        if schema.nullable or not check_obj.isnull().any():
            return CoreCheckResult(passed=True, check="nullable")
        return CoreCheckResult(
            passed=False,
            check="nullable",
            reason_code=(SchemaErrorReason.SERIES_CONTAINS_NULLS),
            message=("non-nullable DataArray contains null values"),
            failure_cases="null",
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_attrs(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check attribute values (equality, regex, callable, or pydantic)."""
        results: list[CoreCheckResult] = []
        if not schema.attrs:
            return results

        if _is_pydantic_model_class(schema.attrs):
            return _validate_attrs_with_pydantic(
                schema.attrs,  # type: ignore[arg-type]
                dict(check_obj.attrs),
            )

        for ak, av in schema.attrs.items():
            if ak not in check_obj.attrs:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(f"missing attribute {ak!r}: expected {av!r}"),
                        failure_cases="<missing>",
                    )
                )
            elif not _match_attr_value(av, check_obj.attrs[ak]):
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"attribute mismatch "
                            f"{ak!r}: expected {av!r}"
                            f", got "
                            f"{check_obj.attrs[ak]!r}"
                        ),
                        failure_cases=str(check_obj.attrs[ak]),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_strict_attrs(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check for unexpected attributes."""
        results: list[CoreCheckResult] = []
        if not (schema.strict_attrs and schema.attrs is not None):
            return results
        if _is_pydantic_model_class(schema.attrs):
            allowed = set(
                schema.attrs.model_fields.keys()  # type: ignore[union-attr]
            )
        else:
            allowed = set(schema.attrs.keys())
        for k in check_obj.attrs:
            if k not in allowed:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="strict_attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(f"unexpected attribute {k!r}"),
                        failure_cases=k,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_coords(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check coordinate presence and sub-schemas."""
        results: list[CoreCheckResult] = []
        if schema.coords is None:
            return results
        if isinstance(schema.coords, list):
            for cn in schema.coords:
                if cn not in check_obj.coords:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="coords",
                            reason_code=(
                                SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
                            ),
                            message=(f"missing coordinate {cn!r}"),
                            failure_cases=cn,
                        )
                    )
        else:
            for cn, cspec in schema.coords.items():
                coord_required = getattr(cspec, "required", True)
                if cn not in check_obj.coords:
                    if coord_required:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=(
                                    SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
                                ),
                                message=(f"missing coordinate {cn!r}"),
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
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_strict_coords(
        self, check_obj, schema: DataArraySchema
    ) -> list[CoreCheckResult]:
        """Check for unexpected coordinates."""
        results: list[CoreCheckResult] = []
        if not (schema.strict_coords and schema.coords is not None):
            return results
        if isinstance(schema.coords, list):
            expected: set[str] = set(schema.coords)
        else:
            expected = set(schema.coords.keys())
        for ck in check_obj.coords:
            if ck not in expected:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="strict_coords",
                        reason_code=(SchemaErrorReason.COLUMN_NOT_IN_SCHEMA),
                        message=(f"unexpected coordinate {ck!r}"),
                        failure_cases=ck,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"coordinate {coord_name!r} "
                            f"must be a dimension "
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
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"coordinate {coord_name!r} "
                            f"must not be a "
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
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"coordinate {coord_name!r} "
                            f"expected indexed on "
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
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"coordinate {coord_name!r} "
                            f"expected non-indexed "
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
                    reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                    message=(f"invalid coordinate spec type: {type(spec)!r}"),
                    failure_cases=str(spec),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

        _run_core_checks(
            error_handler,
            schema,
            check_obj,
            [
                (
                    self.check_name,
                    (check_obj, schema),
                ),
                (
                    self.check_dims,
                    (check_obj, schema),
                ),
                (
                    self.check_sizes,
                    (check_obj, schema),
                ),
                (
                    self.check_shape,
                    (check_obj, schema),
                ),
                (
                    self.check_dtype,
                    (check_obj, schema),
                ),
                (
                    self.check_chunked,
                    (check_obj, schema),
                ),
                (
                    self.check_array_type,
                    (check_obj, schema),
                ),
                (
                    self.check_nullable,
                    (check_obj, schema),
                ),
                (
                    self.check_attrs,
                    (check_obj, schema),
                ),
                (
                    self.check_strict_attrs,
                    (check_obj, schema),
                ),
                (
                    self.check_coords,
                    (check_obj, schema),
                ),
                (
                    self.check_strict_coords,
                    (check_obj, schema),
                ),
                (self.run_checks, (samp, schema)),
            ],
        )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )
        return check_obj


# -------------------------------------------------------------------
# Dataset backend
# -------------------------------------------------------------------


class DatasetSchemaBackend(XarraySchemaBackend):
    """Validate :class:`~xarray.Dataset` against
    :class:`DatasetSchema`."""

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj if inplace else copy.copy(check_obj)

    # ------------------------------------------------------------------
    # Checks — dataset-level structural
    # ------------------------------------------------------------------

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dims(self, ds, schema: DatasetSchema) -> list[CoreCheckResult]:
        """Check dataset-level dimension names."""
        results: list[CoreCheckResult] = []
        if schema.dims is None:
            return results
        ordered = getattr(schema, "ordered_dims", True)
        if ordered:
            exp_tuple = schema.dims
            got_tuple = tuple(ds.dims)
            if exp_tuple != got_tuple:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dims",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"expected dims {exp_tuple!r}, got {got_tuple!r}"
                        ),
                        failure_cases=str(got_tuple),
                    )
                )
        else:
            exp_set = set(schema.dims)
            got_set = set(ds.dims)
            if exp_set != got_set:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="dims",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"expected dims {exp_set!r}, got {got_set!r}"
                        ),
                        failure_cases=str(got_set),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_sizes(self, ds, schema: DatasetSchema) -> list[CoreCheckResult]:
        """Check dataset-level dimension sizes."""
        results: list[CoreCheckResult] = []
        if not schema.sizes:
            return results
        for d, sz in schema.sizes.items():
            if sz is None:
                continue
            if d not in ds.sizes or ds.sizes[d] != sz:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="sizes",
                        reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                        message=(
                            f"dataset size {d!r} "
                            f"expected {sz}, "
                            f"got {ds.sizes.get(d)}"
                        ),
                        failure_cases=str(ds.sizes),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_attrs(self, ds, schema: DatasetSchema) -> list[CoreCheckResult]:
        """Check dataset-level attribute values."""
        results: list[CoreCheckResult] = []
        if not schema.attrs:
            return results

        if _is_pydantic_model_class(schema.attrs):
            return _validate_attrs_with_pydantic(
                schema.attrs,  # type: ignore[arg-type]
                dict(ds.attrs),
                prefix="dataset ",
            )

        for ak, av in schema.attrs.items():
            if ak not in ds.attrs:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"missing dataset attribute "
                            f"{ak!r}: expected {av!r}"
                        ),
                        failure_cases="<missing>",
                    )
                )
            elif not _match_attr_value(av, ds.attrs[ak]):
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"dataset attribute "
                            f"{ak!r}: expected {av!r}"
                            f", got "
                            f"{ds.attrs[ak]!r}"
                        ),
                        failure_cases=str(ds.attrs[ak]),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_strict_attrs(
        self, ds, schema: DatasetSchema
    ) -> list[CoreCheckResult]:
        """Check for unexpected dataset attributes."""
        results: list[CoreCheckResult] = []
        if not (schema.strict_attrs and schema.attrs is not None):
            return results
        if _is_pydantic_model_class(schema.attrs):
            allowed = set(
                schema.attrs.model_fields.keys()  # type: ignore[union-attr]
            )
        else:
            allowed = set(schema.attrs.keys())
        for k in ds.attrs:
            if k not in allowed:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="strict_attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(f"unexpected attribute {k!r}"),
                        failure_cases=k,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_coords(self, ds, schema: DatasetSchema) -> list[CoreCheckResult]:
        """Check coordinate presence and sub-schemas."""
        results: list[CoreCheckResult] = []
        if schema.coords is None:
            return results
        if isinstance(schema.coords, list):
            for cn in schema.coords:
                if cn not in ds.coords:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="coords",
                            reason_code=(
                                SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
                            ),
                            message=(f"missing coordinate {cn!r}"),
                            failure_cases=cn,
                        )
                    )
        else:
            da_backend = DataArraySchemaBackend()
            for cn, cspec in schema.coords.items():
                coord_required = getattr(cspec, "required", True)
                if cn not in ds.coords:
                    if coord_required:
                        results.append(
                            CoreCheckResult(
                                passed=False,
                                check="coords",
                                reason_code=(
                                    SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
                                ),
                                message=(f"missing coordinate {cn!r}"),
                                failure_cases=cn,
                            )
                        )
                elif isinstance(cspec, Coordinate):
                    results.extend(
                        da_backend._validate_coord_on_parent(
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
                            reason_code=(
                                SchemaErrorReason.SCHEMA_COMPONENT_CHECK
                            ),
                            message=(f"invalid coord spec: {type(cspec)!r}"),
                            failure_cases=str(cspec),
                        )
                    )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_strict_coords(
        self, ds, schema: DatasetSchema
    ) -> list[CoreCheckResult]:
        """Check for unexpected coordinates."""
        results: list[CoreCheckResult] = []
        if not (schema.coords is not None and schema.strict_coords):
            return results
        if isinstance(schema.coords, dict):
            allowed = set(schema.coords.keys())
        else:
            allowed = set(schema.coords)
        for ck in ds.coords:
            if ck not in allowed:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="strict_coords",
                        reason_code=(SchemaErrorReason.COLUMN_NOT_IN_SCHEMA),
                        message=(f"unexpected coordinate {ck!r}"),
                        failure_cases=ck,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Checks — data-variable level
    # ------------------------------------------------------------------

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_strict_data_vars(
        self,
        schema: DatasetSchema,
        extras: list,
    ) -> CoreCheckResult:
        """Fail when ``strict=True`` and extra vars exist.

        *extras* is pre-computed before the strict filter so that
        ``strict="filter"`` and ``strict=True`` see the same set.
        """
        if schema.strict is not True or not extras:
            return CoreCheckResult(passed=True, check="strict_data_vars")
        return CoreCheckResult(
            passed=False,
            check="strict_data_vars",
            reason_code=(SchemaErrorReason.COLUMN_NOT_IN_SCHEMA),
            message=(f"unexpected data variables: {extras}"),
            failure_cases=str(extras),
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_data_var_presence(
        self,
        ds,
        schema: DatasetSchema,
        logical_to_actual: dict[str, str],
    ) -> list[CoreCheckResult]:
        """Check that required data variables are present."""
        results: list[CoreCheckResult] = []
        for logical, spec in schema.data_vars.items():
            actual = logical_to_actual[logical]
            if actual in ds.data_vars:
                continue
            required = True
            if isinstance(spec, DataVar):
                required = spec.required
            if required:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="data_var_presence",
                        reason_code=(
                            SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
                        ),
                        message=(f"missing required data_var {actual!r}"),
                        failure_cases=actual,
                    )
                )
        return results

    def run_schema_component_checks(  # type: ignore[override]
        self,
        ds,
        schema: DatasetSchema,
        logical_to_actual: dict[str, str],
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
    ) -> list[CoreCheckResult]:
        """Validate each present data variable against its
        sub-schema.

        Not scope-gated: delegates to per-variable
        ``da_backend.validate()`` which applies its own
        ``@validate_scope`` checks internally.
        """
        results: list[CoreCheckResult] = []
        da_backend = DataArraySchemaBackend()
        for logical, spec in schema.data_vars.items():
            actual = logical_to_actual[logical]
            if actual not in ds.data_vars or spec is None:
                continue
            if isinstance(spec, DataArraySchema):
                sub = copy.copy(spec)
            else:
                sub = copy.copy(spec.to_data_array_schema(logical))
            sub = sub.set_name(actual)
            try:
                da_backend.validate(
                    ds[actual],
                    sub,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except SchemaErrors as exc:
                for e in exc.schema_errors:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check=actual,
                            reason_code=e.reason_code,
                            schema_error=e,
                        )
                    )
            except SchemaError as e:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check=actual,
                        reason_code=e.reason_code,
                        schema_error=e,
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_data_var_alignment(
        self,
        ds,
        schema: DatasetSchema,
        logical_to_actual: dict[str, str],
    ) -> list[CoreCheckResult]:
        """Check aligned_with / broadcastable_with."""
        results: list[CoreCheckResult] = []
        for logical, spec in schema.data_vars.items():
            if not isinstance(spec, DataVar):
                continue
            actual = logical_to_actual[logical]
            if actual not in ds.data_vars:
                continue
            da_self = ds[actual]
            for other_log in spec.aligned_with or ():
                other = logical_to_actual.get(other_log, other_log)
                if other not in ds.data_vars:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="aligned_with",
                            reason_code=(
                                SchemaErrorReason.SCHEMA_COMPONENT_CHECK
                            ),
                            message=(
                                f"aligned_with: peer "
                                f"{other_log!r} "
                                f"({other!r}) missing"
                            ),
                        )
                    )
                    continue
                if not _aligned_dims_sizes(da_self, ds[other]):
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="aligned_with",
                            reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                            message=(
                                f"{actual!r} not aligned"
                                f" with {other!r} "
                                f"(dims/sizes)"
                            ),
                        )
                    )
            for other_log in spec.broadcastable_with or ():
                other = logical_to_actual.get(other_log, other_log)
                if other not in ds.data_vars:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="broadcastable_with",
                            reason_code=(
                                SchemaErrorReason.SCHEMA_COMPONENT_CHECK
                            ),
                            message=(
                                f"broadcastable_with: "
                                f"peer {other_log!r} "
                                f"missing"
                            ),
                        )
                    )
                    continue
                if not _broadcast_compatible(da_self, ds[other]):
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="broadcastable_with",
                            reason_code=(SchemaErrorReason.MISMATCH_INDEX),
                            message=(
                                f"{actual!r} not "
                                f"broadcast-compatible "
                                f"with {other!r}"
                            ),
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_var_name(
        self,
        logical: str,
        spec: DataVar | DataArraySchema | None,
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
                    "inserting scalar default requires "
                    f"DataVar.dims for {actual_name!r}"
                ),
                reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
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
                    message=(f"cannot fill default: dim {d!r} not on dataset"),
                    reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                )
            shape.append(ds.sizes[d])
            if d in ds.coords:
                coords[d] = ds.coords[d]
        arr = np.full(shape, spec.default)
        da = xr.DataArray(
            arr,
            dims=[d for d in spec.dims if d is not None],
            coords=coords,
        )
        return ds.assign(**{actual_name: da})

    def _fill_data_var_defaults(
        self,
        ds: Any,
        schema: DatasetSchema,
        logical_to_actual: dict[str, str],
        error_handler: ErrorHandler,
    ) -> Any:
        """Fill default values for missing optional vars."""
        for logical, spec in schema.data_vars.items():
            if not isinstance(spec, DataVar):
                continue
            actual = logical_to_actual[logical]
            if actual in ds.data_vars:
                continue
            if spec.required or spec.default is None:
                continue
            try:
                ds = self._apply_default(ds, actual, spec, schema)
            except SchemaError as exc:
                error_handler.collect_error(
                    get_error_category(exc.reason_code),
                    exc.reason_code,
                    exc,
                )
        return ds

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

        # --- resolve logical → actual var names ---
        logical_to_actual: dict[str, str] = {}
        for logical, spec in schema.data_vars.items():
            logical_to_actual[logical] = self._resolve_var_name(logical, spec)

        # --- fail fast: duplicate alias resolution ---
        actual_to_logicals: dict[str, list[str]] = {}
        for logical, actual in logical_to_actual.items():
            actual_to_logicals.setdefault(actual, []).append(logical)
        dupes = {
            a: lgs for a, lgs in actual_to_logicals.items() if len(lgs) > 1
        }
        if dupes:
            detail = ", ".join(f"{a!r} <- {lgs}" for a, lgs in dupes.items())
            raise SchemaError(
                schema,
                data=ds,
                message=(
                    "multiple data_vars resolve to the "
                    "same actual variable name: "
                    f"{detail}"
                ),
                reason_code=SchemaErrorReason.DUPLICATES,
            )

        # --- strict filter (preprocessing) ---
        planned = {logical_to_actual[k] for k in schema.data_vars}
        extras = [v for v in ds.data_vars if v not in planned]
        if schema.strict == "filter" and extras:
            ds = ds.drop_vars(extras)

        # --- parsers ---
        try:
            ds = self.run_parsers(schema, ds)
        except SchemaError as exc:
            error_handler.collect_error(
                get_error_category(exc.reason_code),
                exc.reason_code,
                exc,
            )

        # --- fill defaults for optional vars ---
        ds = self._fill_data_var_defaults(
            ds, schema, logical_to_actual, error_handler
        )

        # --- run all checks ---
        _run_core_checks(
            error_handler,
            schema,
            ds,
            [
                (
                    self.check_strict_data_vars,
                    (schema, extras),
                ),
                (self.check_dims, (ds, schema)),
                (self.check_sizes, (ds, schema)),
                (self.check_attrs, (ds, schema)),
                (
                    self.check_strict_attrs,
                    (ds, schema),
                ),
                (self.check_coords, (ds, schema)),
                (
                    self.check_strict_coords,
                    (ds, schema),
                ),
                (
                    self.check_data_var_presence,
                    (ds, schema, logical_to_actual),
                ),
                (
                    self.run_schema_component_checks,
                    (
                        ds,
                        schema,
                        logical_to_actual,
                        head,
                        tail,
                        sample,
                        random_state,
                        lazy,
                    ),
                ),
                (
                    self.check_data_var_alignment,
                    (ds, schema, logical_to_actual),
                ),
                (self.run_checks, (ds, schema)),
            ],
        )

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=ds,
            )
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"expected Dataset, got {type(ds)}")
        return ds
