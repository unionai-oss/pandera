"""Generate synthetic xarray data from schema definitions.

This module generates :class:`xarray.DataArray` and :class:`xarray.Dataset`
objects that conform to :class:`~pandera.api.xarray.container.DataArraySchema`
and :class:`~pandera.api.xarray.container.DatasetSchema` specifications.

Built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_
package.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np

from pandera.engines import numpy_engine
from pandera.engines.xarray_engine import DataType as XarrayDataType
from pandera.errors import SchemaDefinitionError
from pandera.strategies.base_strategies import (
    CONSTRAINT_DISPATCHER,
    HAS_HYPOTHESIS,
)
from pandera.strategies.constraints import (
    UNSET,
    ConstraintConflictError,
    FieldConstraints,
)

if HAS_HYPOTHESIS:
    import hypothesis.extra.numpy as npst
    import hypothesis.strategies as st
    from hypothesis.strategies import SearchStrategy, composite
else:  # pragma: no cover
    from pandera.strategies.base_strategies import SearchStrategy, composite

F = TypeVar("F")

DEFAULT_DIM_SIZE = 3
DEFAULT_DIMS = ("dim_0",)


def _strategy_import_error(fn: F) -> F:
    """Decorator to raise ImportError when hypothesis is missing."""

    @wraps(fn)  # type: ignore[arg-type]
    def _wrapper(*args, **kwargs):
        if not HAS_HYPOTHESIS:  # pragma: no cover
            raise ImportError(
                'Strategies for generating data requires "hypothesis" '
                "to be installed.\n"
                "pip install pandera[strategies]"
            )
        return fn(*args, **kwargs)

    # @composite sets __qualname__ to accept.<locals>.<name>. Sphinx autodoc
    # uses inspect.unwrap() and then treats those as local functions. Normalize
    # every function in the __wrapped__ chain (including the wrapper from @wraps).
    _cur: Any = _wrapper
    while _cur is not None:
        if "<locals>" in getattr(_cur, "__qualname__", ""):
            _cur.__qualname__ = _cur.__name__
        _cur = getattr(_cur, "__wrapped__", None)

    return cast(F, _wrapper)


def _to_numpy_dtype(dtype_spec) -> np.dtype:
    """Resolve a dtype specification to a numpy dtype."""
    if dtype_spec is None:
        return np.dtype("float64")
    if isinstance(dtype_spec, np.dtype):
        return dtype_spec
    if isinstance(dtype_spec, XarrayDataType):
        return dtype_spec.type
    try:
        return np.dtype(dtype_spec)
    except TypeError:
        return np.dtype("float64")


def _aggregate_constraints(
    checks: Sequence | None,
    dispatch_type: type,
) -> tuple[FieldConstraints, list]:
    """Bucket xarray-level checks into constraint adapters and residuals.

    Mirrors the bucketing in
    :func:`pandera.strategies.pandas_strategies.field_element_strategy`
    so xarray strategies can leverage the same ``FieldConstraints``
    aggregation machinery (per ``specs/optimized-strategies.md`` §5).

    :param checks: sequence of ``Check`` objects to aggregate.
    :param dispatch_type: the dispatch key (``xr.DataArray`` for
        ``data_array_strategy`` or ``xr.Dataset`` for
        ``dataset_strategy``) used to look up adapters in
        :data:`CONSTRAINT_DISPATCHER`.
    :returns: ``(merged_constraints, leftover_checks)`` — checks that
        did not contribute a constraint adapter remain in the leftover
        list so callers can decide whether to warn or filter.
    :raises SchemaDefinitionError: when the merged constraints are
        jointly unsatisfiable.
    """
    constraint_acc = FieldConstraints()
    leftover: list = []
    for check in checks or ():
        constraint_fn = getattr(check, "constraint", None) or (
            CONSTRAINT_DISPATCHER.get((check.name, dispatch_type))
        )
        if constraint_fn is None:
            leftover.append(check)
            continue
        try:
            constraint_acc = constraint_acc.merge(
                constraint_fn(**check.statistics)
            )
        except ConstraintConflictError as exc:
            names = [str(getattr(c, "name", "?")) for c in checks or ()]
            raise SchemaDefinitionError(
                f"Cannot construct a data-generation strategy for checks "
                f"{names}: constraints are jointly unsatisfiable ({exc})."
            ) from exc
    return constraint_acc, leftover


def _close_bound_for_npdtype(
    np_dtype: np.dtype,
    value: Any,
    exclude: bool,
    side: str,
) -> tuple[Any, bool]:
    """Adjust an exclusive bound for a numpy dtype.

    ``hypothesis.extra.numpy.from_dtype`` natively supports exclusive
    bounds for floats / complex dtypes via ``exclude_min`` /
    ``exclude_max`` but treats integer bounds as inclusive. For
    integer-like dtypes we close the interval by adjusting by 1
    (mirrors ``pandas_strategies._close_bound``).
    """
    if np.issubdtype(np_dtype, np.floating) or np.issubdtype(
        np_dtype, np.complexfloating
    ):
        return value, exclude
    if not exclude:
        return value, False
    if np.issubdtype(np_dtype, np.integer):
        try:
            return (value + 1) if side == "min" else (value - 1), False
        except TypeError:
            return value, False
    return value, False


@_strategy_import_error
def compile_dataarray_element_strategy(
    np_dtype: np.dtype,
    constraints: FieldConstraints,
) -> SearchStrategy:
    """Lower a merged ``FieldConstraints`` to a single element strategy.

    Analogous to
    :func:`pandera.strategies.pandas_strategies.compile_field_strategy`,
    but emits ``hypothesis.extra.numpy.from_dtype`` instead of the
    pandas dtype strategy. Equality, ``isin``, ``notin`` and
    numeric-bound aggregation are honoured; string aggregation falls
    back to ``st.text`` since ``npst.from_dtype`` doesn't model
    regex-shaped strings directly.
    """
    constraints = constraints.apply_post_merge_hooks()

    if constraints.eq is not UNSET:
        if constraints.eq in constraints.notin:
            raise ConstraintConflictError(
                f"eq={constraints.eq!r} conflicts with notin"
            )
        return st.just(constraints.eq).map(np_dtype.type)

    if constraints.isin is not None:
        allowed = set(constraints.isin) - set(constraints.notin)
        if constraints.min_value is not UNSET:
            cmp = operator.lt if constraints.exclude_min else operator.le
            allowed = {v for v in allowed if cmp(constraints.min_value, v)}
        if constraints.max_value is not UNSET:
            cmp = operator.gt if constraints.exclude_max else operator.ge
            allowed = {v for v in allowed if cmp(constraints.max_value, v)}
        if not allowed:
            raise ConstraintConflictError(
                "isin/notin/bounds intersection is empty"
            )
        strat = st.sampled_from(sorted(allowed)).map(np_dtype.type)
        for _name, predicate in constraints.residual_filters:
            strat = strat.filter(predicate)
        return strat

    is_string = np_dtype.kind in {"U", "S", "O"}
    if is_string and (
        constraints.regex_fullmatch
        or constraints.regex_search
        or constraints.str_min_len is not None
        or constraints.str_max_len is not None
        or constraints.str_exact_len is not None
    ):
        # Reuse the pandas string compiler — the constraints layer is
        # dtype-agnostic; we just need a string-shaped element strategy.
        # A circular import is avoided by a local import.
        from pandera.engines import pandas_engine
        from pandera.strategies.pandas_strategies import (
            _compile_string_strategy,
        )

        return _compile_string_strategy(
            pandas_engine.Engine.dtype(str), constraints
        )

    kwargs: dict[str, Any] = {}
    if constraints.min_value is not UNSET:
        min_v, excl_min = _close_bound_for_npdtype(
            np_dtype, constraints.min_value, constraints.exclude_min, "min"
        )
        kwargs["min_value"] = min_v
        if np.issubdtype(np_dtype, np.floating) or np.issubdtype(
            np_dtype, np.complexfloating
        ):
            kwargs["exclude_min"] = excl_min
    if constraints.max_value is not UNSET:
        max_v, excl_max = _close_bound_for_npdtype(
            np_dtype, constraints.max_value, constraints.exclude_max, "max"
        )
        kwargs["max_value"] = max_v
        if np.issubdtype(np_dtype, np.floating) or np.issubdtype(
            np_dtype, np.complexfloating
        ):
            kwargs["exclude_max"] = excl_max
    if np.issubdtype(np_dtype, np.floating) or np.issubdtype(
        np_dtype, np.complexfloating
    ):
        kwargs["allow_nan"] = constraints.allow_nan
        kwargs["allow_infinity"] = constraints.allow_infinity

    if np.issubdtype(np_dtype, np.datetime64):
        strat = npst.from_dtype(
            np.dtype("int64"),
            **{"allow_nan": False, "allow_infinity": False, **kwargs},
        ).map(lambda x: np.datetime64(x, "ns"))
    elif np.issubdtype(np_dtype, np.timedelta64):
        strat = npst.from_dtype(
            np.dtype("int64"),
            **{"allow_nan": False, "allow_infinity": False, **kwargs},
        ).map(lambda x: np.timedelta64(x, "ns"))
    else:
        strat = npst.from_dtype(
            np_dtype,
            **{"allow_nan": False, "allow_infinity": False, **kwargs},
        )

    if constraints.notin:
        forbidden = constraints.notin
        strat = strat.filter(lambda v, f=forbidden: v not in f)

    for _name, predicate in constraints.residual_filters:
        strat = strat.filter(predicate)

    return strat


def _checks_to_element_strategy(
    np_dtype: np.dtype,
    dtype_spec: Any,
    checks: Sequence | None,
    dispatch_type: type,
) -> tuple[SearchStrategy, list]:
    """Lower xarray-level checks to a single hypothesis element strategy.

    Returns the element strategy plus any leftover checks (those
    without a constraint adapter). The xarray strategy layer does not
    currently fall back to ``.filter`` chaining for leftover checks —
    it ignores them and lets the schema validate-time pass surface
    failures, which matches the pre-Stage-8 behaviour. A future
    revision may surface a ``DeprecationWarning`` analogous to
    ``pandas_strategies._warn_legacy_strategy_chained_once``.
    """
    constraints, leftover = _aggregate_constraints(checks, dispatch_type)
    if constraints.is_empty():
        elements = xarray_dtype_strategy(dtype_spec)
    else:
        elements = compile_dataarray_element_strategy(np_dtype, constraints)
    return elements, leftover


@_strategy_import_error
def xarray_dtype_strategy(
    dtype_spec,
    strategy: SearchStrategy | None = None,
    **kwargs,
) -> SearchStrategy:
    """Strategy to generate scalar values matching a numpy dtype.

    :param dtype_spec: numpy dtype or pandera DataType.
    :param strategy: optional strategy to chain onto.
    :param kwargs: passed to ``hypothesis.extra.numpy.from_dtype``.
    :returns: hypothesis strategy producing scalar values.
    """
    np_dtype = _to_numpy_dtype(dtype_spec)

    if strategy is not None:
        return strategy.map(np_dtype.type)

    if np.issubdtype(np_dtype, np.datetime64):
        return npst.from_dtype(
            np.dtype("int64"),
            **{"allow_nan": False, "allow_infinity": False, **kwargs},
        ).map(lambda x: np.datetime64(x, "ns"))

    if np.issubdtype(np_dtype, np.timedelta64):
        return npst.from_dtype(
            np.dtype("int64"),
            **{"allow_nan": False, "allow_infinity": False, **kwargs},
        ).map(lambda x: np.timedelta64(x, "ns"))

    return npst.from_dtype(
        np_dtype,
        **{"allow_nan": False, "allow_infinity": False, **kwargs},
    )


@composite
def _numpy_array_strategy(
    draw,
    dtype_spec,
    shape: tuple[int, ...],
    element_strategy: SearchStrategy | None = None,
):
    """Strategy producing a numpy array of a given dtype and shape."""
    np_dtype = _to_numpy_dtype(dtype_spec)
    if element_strategy is None:
        element_strategy = xarray_dtype_strategy(dtype_spec)
    arr = draw(
        npst.arrays(dtype=np_dtype, shape=shape, elements=element_strategy)
    )
    return arr


@_strategy_import_error
@composite
def data_array_strategy(
    draw,
    dtype=None,
    dims: tuple[str, ...] | None = None,
    sizes: dict[str, int | None] | None = None,
    shape: tuple[int | None, ...] | None = None,
    coords: dict | None = None,
    name: str | None = None,
    checks: Sequence | None = None,
    nullable: bool = False,
    size: int | None = None,
):
    """Strategy to generate an :class:`xarray.DataArray`.

    :param dtype: numpy dtype or string for the array data.
    :param dims: dimension names.
    :param sizes: mapping of dim name to size.
    :param shape: positional shape tuple.
    :param coords: mapping of coord name to values/strategy.
    :param name: name for the DataArray.
    :param checks: optional sequence of :class:`~pandera.api.checks.Check`
        instances. Built-in checks with constraint adapters compose
        into a single ``hypothesis`` element strategy so generated
        values satisfy the constraints by construction (per
        ``specs/optimized-strategies.md`` §5).
    :param nullable: if True, sprinkle NaN values.
    :param size: default size for dimensions without an explicit size.
    :returns: hypothesis strategy producing ``xr.DataArray``.
    """
    import xarray as xr

    default_size = size or DEFAULT_DIM_SIZE
    if dims is None:
        dims = DEFAULT_DIMS

    resolved_shape = _resolve_shape(dims, sizes, shape, default_size)

    np_dtype = _to_numpy_dtype(dtype)
    element_strat, _leftover = _checks_to_element_strategy(
        np_dtype, dtype, checks, xr.DataArray
    )
    data = draw(
        npst.arrays(
            dtype=np_dtype, shape=resolved_shape, elements=element_strat
        )
    )

    if nullable and np.issubdtype(np_dtype, np.floating):
        mask = draw(
            npst.arrays(
                dtype=np.bool_,
                shape=resolved_shape,
                elements=st.booleans(),
            )
        )
        data = np.where(mask, np.nan, data)

    coord_dict = _build_coords(draw, dims, resolved_shape, coords)

    return xr.DataArray(
        data=data,
        dims=dims,
        coords=coord_dict if coord_dict else None,
        name=name,
    )


@_strategy_import_error
@composite
def dataset_strategy(
    draw,
    data_vars: dict[str, dict[str, Any]] | None = None,
    coords: dict | None = None,
    dims: tuple[str, ...] | None = None,
    sizes: dict[str, int | None] | None = None,
    size: int | None = None,
):
    """Strategy to generate an :class:`xarray.Dataset`.

    :param data_vars: mapping of var name to dict with ``dtype``, ``dims``,
        ``nullable``, etc.
    :param coords: mapping of coord name to coord spec.
    :param dims: dataset-level dimension names.
    :param sizes: mapping of dim name to size.
    :param size: default size for dimensions.
    :returns: hypothesis strategy producing ``xr.Dataset``.
    """
    import xarray as xr

    default_size = size or DEFAULT_DIM_SIZE

    if data_vars is None:
        data_vars = {
            "var_0": {"dtype": "float64", "dims": DEFAULT_DIMS},
        }

    all_dims: set[str] = set()
    for spec in data_vars.values():
        var_dims = spec.get("dims", DEFAULT_DIMS)
        all_dims.update(var_dims)

    dim_sizes = _resolve_dim_sizes(all_dims, sizes, default_size)

    coord_dict: dict[str, Any] = {}
    if coords:
        for coord_name, coord_spec in coords.items():
            if isinstance(coord_spec, dict):
                coord_dtype = coord_spec.get("dtype", "float64")
                if coord_name in dim_sizes:
                    coord_len = dim_sizes[coord_name]
                    coord_data = draw(
                        npst.arrays(
                            dtype=_to_numpy_dtype(coord_dtype),
                            shape=(coord_len,),
                            elements=xarray_dtype_strategy(coord_dtype),
                        )
                    )
                    coord_dict[coord_name] = (coord_name, coord_data)
            elif coord_name in dim_sizes:
                coord_len = dim_sizes[coord_name]
                coord_data = draw(
                    npst.arrays(
                        dtype=np.float64,
                        shape=(coord_len,),
                        elements=xarray_dtype_strategy("float64"),
                    )
                )
                coord_dict[coord_name] = (coord_name, coord_data)

    ds_vars: dict[str, Any] = {}
    for var_name, spec in data_vars.items():
        var_dims = spec.get("dims", DEFAULT_DIMS)
        var_dtype = spec.get("dtype", "float64")
        var_nullable = spec.get("nullable", False)
        var_checks = spec.get("checks")
        var_shape = tuple(dim_sizes.get(d, default_size) for d in var_dims)

        np_dtype = _to_numpy_dtype(var_dtype)
        var_elements, _leftover = _checks_to_element_strategy(
            np_dtype, var_dtype, var_checks, xr.DataArray
        )
        arr = draw(
            npst.arrays(
                dtype=np_dtype,
                shape=var_shape,
                elements=var_elements,
            )
        )

        if var_nullable and np.issubdtype(np_dtype, np.floating):
            mask = draw(
                npst.arrays(
                    dtype=np.bool_,
                    shape=var_shape,
                    elements=st.booleans(),
                )
            )
            arr = np.where(mask, np.nan, arr)

        ds_vars[var_name] = (var_dims, arr)

    return xr.Dataset(ds_vars, coords=coord_dict if coord_dict else None)


@_strategy_import_error
def data_array_schema_strategy(
    schema,
    size: int | None = None,
) -> SearchStrategy:
    """Create a strategy from a DataArraySchema.

    :param schema: the DataArraySchema to use.
    :param size: default dimension size.
    :returns: hypothesis strategy producing conforming DataArrays.
    """
    coord_specs = None
    if schema.coords:
        from pandera.api.xarray.components import Coordinate

        coord_specs = {}
        if isinstance(schema.coords, list):
            for name in schema.coords:
                coord_specs[name] = {"dtype": "float64"}
        else:
            for name, coord in schema.coords.items():
                if isinstance(coord, Coordinate):
                    coord_specs[name] = {
                        "dtype": str(coord.dtype) if coord.dtype else "float64"
                    }
                else:
                    coord_specs[name] = {"dtype": "float64"}

    return data_array_strategy(
        dtype=schema.dtype,
        dims=schema.dims,
        sizes=schema.sizes,
        shape=schema.shape,
        coords=coord_specs,
        name=schema.name,
        checks=schema.checks,
        nullable=schema.nullable,
        size=size,
    )


@_strategy_import_error
def dataset_schema_strategy(
    schema,
    size: int | None = None,
) -> SearchStrategy:
    """Create a strategy from a DatasetSchema.

    :param schema: the DatasetSchema to use.
    :param size: default dimension size.
    :returns: hypothesis strategy producing conforming Datasets.
    """
    from pandera.api.xarray.components import Coordinate, DataVar

    dv_specs: dict[str, dict[str, Any]] = {}
    if schema.data_vars:
        for key, spec in schema.data_vars.items():
            if spec is None:
                dv_specs[key] = {"dtype": "float64", "dims": DEFAULT_DIMS}
                continue
            if isinstance(spec, DataVar):
                dv_specs[key] = {
                    "dtype": str(spec.dtype) if spec.dtype else "float64",
                    "dims": spec.dims or DEFAULT_DIMS,
                    "nullable": spec.nullable,
                    "checks": spec.checks,
                }
            else:
                dv_specs[key] = {
                    "dtype": str(spec.dtype) if spec.dtype else "float64",
                    "dims": spec.dims or DEFAULT_DIMS,
                    "nullable": getattr(spec, "nullable", False),
                    "checks": getattr(spec, "checks", None),
                }

    coord_specs = None
    if schema.coords:
        coord_specs = {}
        if isinstance(schema.coords, list):
            for name in schema.coords:
                coord_specs[name] = {"dtype": "float64"}
        else:
            for name, coord in schema.coords.items():
                if isinstance(coord, Coordinate):
                    coord_specs[name] = {
                        "dtype": str(coord.dtype) if coord.dtype else "float64"
                    }
                else:
                    coord_specs[name] = {"dtype": "float64"}

    return dataset_strategy(
        data_vars=dv_specs if dv_specs else None,
        coords=coord_specs,
        dims=schema.dims,
        sizes=schema.sizes,
        size=size,
    )


def _resolve_shape(
    dims: tuple[str, ...],
    sizes: dict[str, int | None] | None,
    shape: tuple[int | None, ...] | None,
    default_size: int,
) -> tuple[int, ...]:
    """Resolve dimension lengths from sizes/shape/default."""
    if shape is not None:
        return tuple(s if s is not None else default_size for s in shape)
    if sizes is not None:
        return tuple(sizes.get(d, default_size) or default_size for d in dims)
    return tuple(default_size for _ in dims)


def _resolve_dim_sizes(
    all_dims: set[str],
    sizes: dict[str, int | None] | None,
    default_size: int,
) -> dict[str, int]:
    """Resolve dimension sizes for all dimensions."""
    result: dict[str, int] = {}
    for d in sorted(all_dims):
        if sizes and d in sizes:
            result[d] = sizes[d] or default_size
        else:
            result[d] = default_size
    return result


def _build_coords(
    draw,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    coords: dict | None,
) -> dict[str, Any]:
    """Build coordinate arrays for a DataArray strategy."""
    import xarray as xr

    coord_dict: dict[str, Any] = {}
    if coords is None:
        return coord_dict

    for coord_name, coord_spec in coords.items():
        if isinstance(coord_spec, dict):
            coord_dtype = coord_spec.get("dtype", "float64")
        else:
            coord_dtype = "float64"

        if coord_name in dims:
            dim_idx = dims.index(coord_name)
            coord_len = shape[dim_idx]
            np_dtype = _to_numpy_dtype(coord_dtype)
            coord_data = draw(
                npst.arrays(
                    dtype=np_dtype,
                    shape=(coord_len,),
                    elements=xarray_dtype_strategy(coord_dtype),
                )
            )
            coord_dict[coord_name] = (coord_name, coord_data)

    return coord_dict
