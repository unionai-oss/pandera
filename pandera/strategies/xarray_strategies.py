"""Generate synthetic xarray data from schema definitions.

This module generates :class:`xarray.DataArray` and :class:`xarray.Dataset`
objects that conform to :class:`~pandera.api.xarray.container.DataArraySchema`
and :class:`~pandera.api.xarray.container.DatasetSchema` specifications.

Built on top of the
`hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_
package.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np

from pandera.engines import numpy_engine
from pandera.engines.xarray_engine import DataType as XarrayDataType
from pandera.errors import SchemaDefinitionError
from pandera.strategies.base_strategies import HAS_HYPOTHESIS

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
    :param checks: (unused) reserved for future check-aware generation.
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
    element_strat = xarray_dtype_strategy(dtype)
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
        var_shape = tuple(dim_sizes.get(d, default_size) for d in var_dims)

        np_dtype = _to_numpy_dtype(var_dtype)
        arr = draw(
            npst.arrays(
                dtype=np_dtype,
                shape=var_shape,
                elements=xarray_dtype_strategy(var_dtype),
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
                }
            else:
                dv_specs[key] = {
                    "dtype": str(spec.dtype) if spec.dtype else "float64",
                    "dims": spec.dims or DEFAULT_DIMS,
                    "nullable": getattr(spec, "nullable", False),
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
