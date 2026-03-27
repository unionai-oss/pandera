"""Xarray-specific built-in :class:`~pandera.api.checks.Check` functions.

Registered implementations dispatch on :class:`xarray.DataArray` or
:class:`xarray.Dataset` (see :data:`XrLike`).

**Dataset schema checks**

For a :class:`~xarray.Dataset`, element-wise builtins (comparisons, ``isin``,
string helpers, etc.) require **every** data variable to satisfy the check
(logical AND). The check returns a single aggregate ``bool``.

**``unique_values_eq``**

On a Dataset, the unique values from **all** data variables are combined
(union); that set must equal the expected set.

**``isin`` / ``notin``**

:class:`~pandera.api.checks.Check` stores membership arguments as a
``frozenset``. xarray's :meth:`~xarray.DataArray.isin` treats a set-like
argument as *one* label, so values are normalized via :func:`_isin_test_elements`.

**Structural checks vs :class:`~pandera.api.xarray.container.DataArraySchema` /
:class:`~pandera.api.xarray.container.DatasetSchema`**

Several builtins mirror constructor arguments (``has_dims`` ↔ ``dims``,
``has_coords`` ↔ coordinate specs, ``has_attrs`` ↔ ``attrs``, ``dim_size`` ↔
``sizes``, ``ndim`` ↔ length of ``dims``). Prefer the schema kwargs when you are
defining a full schema—validation runs in one place and errors stay
schema-centric.

These checks remain useful when: validating with a minimal schema plus ad hoc
constraints; attaching extra structural rules at **dataset** scope
(``DatasetSchema(..., checks=[...])``); composing with data-level checks; or
reusing :class:`~pandera.api.checks.Check` where no container schema exists.

``is_monotonic`` and ``no_duplicates_in_coord`` constrain coordinate **values**
and are not redundant with typical ``dims`` / ``sizes`` fields (unless you encode
the same rule elsewhere, e.g. on a :class:`~pandera.api.xarray.components.Coordinate`).
"""

import operator
import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import numpy as np
import xarray as xr

from pandera.api.extensions import register_builtin_check

XrLike = xr.DataArray | xr.Dataset

T = TypeVar("T")


def _isin_test_elements(values: Iterable) -> Iterable:
    """Normalize values for :meth:`xarray.DataArray.isin`.

    ``isin`` treats ``set`` / ``frozenset`` as a single test label; expand to a
    list so each element is tested for membership (matches pandas semantics).
    """
    if isinstance(values, (set, frozenset)):
        return list(values)
    return values


def _dataset_reduce_bool(
    ds: xr.Dataset, per_da: Callable[[xr.DataArray], Any]
) -> bool:
    """Whether ``per_da`` yields an all-True boolean array for every data var."""
    for _name, da in ds.data_vars.items():
        out = per_da(da)
        if isinstance(out, xr.DataArray):
            if not bool(np.all(out.values)):
                return False
        elif not bool(np.all(out)):
            return False
    return True


def _str_element_unusable(x: Any) -> bool:
    """True for None / NaN — string builtins treat these as failed cells."""
    if x is None:
        return True
    try:
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return True
    except TypeError:
        pass
    return False


def _str_mask_predicate(
    data: xr.DataArray,
    pred: Callable[[str], bool],
) -> xr.DataArray:
    """Apply ``pred`` per element; build a boolean :class:`~xarray.DataArray`.

    Uses :func:`numpy.frompyfunc` so object-dtype arrays (e.g. mixed strings)
    are handled without assuming a vectorized string ufunc exists on the dtype.
    """

    def _scalar(x: Any) -> bool:
        if _str_element_unusable(x):
            return False
        return pred(str(x))

    flat = np.asarray(data.values).ravel()
    out = np.asarray(np.frompyfunc(_scalar, 1, 1)(flat), dtype=bool)
    return xr.DataArray(
        out.reshape(np.asarray(data.values).shape),
        dims=data.dims,
        coords=data.coords,
    )


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: XrLike, value: Any) -> bool | xr.DataArray:
    """Element-wise ``data == value``.

    :param data: DataArray (boolean mask result) or Dataset (all vars checked).
    :param value: Scalar compared to each element (xarray broadcasting rules).
    """
    if isinstance(data, xr.DataArray):
        return data == value
    return _dataset_reduce_bool(data, lambda da: da == value)


@register_builtin_check(
    aliases=["ne"],
    error="not_equal_to({value})",
)
def not_equal_to(data: XrLike, value: Any) -> bool | xr.DataArray:
    """Element-wise ``data != value``.

    :param data: DataArray or Dataset (see :func:`equal_to`).
    :param value: Scalar compared to each element.
    """
    if isinstance(data, xr.DataArray):
        return data != value
    return _dataset_reduce_bool(data, lambda da: da != value)


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({min_value})",
)
def greater_than(data: XrLike, min_value: Any) -> bool | xr.DataArray:
    """Element-wise strict lower bound ``data > min_value``."""
    if isinstance(data, xr.DataArray):
        return data > min_value
    return _dataset_reduce_bool(data, lambda da: da > min_value)


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(
    data: XrLike, min_value: Any
) -> bool | xr.DataArray:
    """Element-wise ``data >= min_value``."""
    if isinstance(data, xr.DataArray):
        return data >= min_value
    return _dataset_reduce_bool(data, lambda da: da >= min_value)


@register_builtin_check(
    aliases=["lt"],
    error="less_than({max_value})",
)
def less_than(data: XrLike, max_value: Any) -> bool | xr.DataArray:
    """Element-wise strict upper bound ``data < max_value``."""
    if max_value is None:
        raise ValueError("max_value must not be None")
    if isinstance(data, xr.DataArray):
        return data < max_value
    return _dataset_reduce_bool(data, lambda da: da < max_value)


@register_builtin_check(
    aliases=["le"],
    error="less_than_or_equal_to({max_value})",
)
def less_than_or_equal_to(data: XrLike, max_value: Any) -> bool | xr.DataArray:
    """Element-wise ``data <= max_value``."""
    if max_value is None:
        raise ValueError("max_value must not be None")
    if isinstance(data, xr.DataArray):
        return data <= max_value
    return _dataset_reduce_bool(data, lambda da: da <= max_value)


@register_builtin_check(
    error="isin({allowed_values})",
)
def isin(data: XrLike, allowed_values: Iterable) -> bool | xr.DataArray:
    """Whether each value appears in ``allowed_values`` (after normalization)."""
    test_elements = _isin_test_elements(allowed_values)
    if isinstance(data, xr.DataArray):
        return data.isin(test_elements)
    return _dataset_reduce_bool(data, lambda da: da.isin(test_elements))


@register_builtin_check(
    error="notin({forbidden_values})",
)
def notin(data: XrLike, forbidden_values: Iterable) -> bool | xr.DataArray:
    """Negation of :func:`isin` (element-wise)."""
    test_elements = _isin_test_elements(forbidden_values)
    if isinstance(data, xr.DataArray):
        return ~data.isin(test_elements)
    return _dataset_reduce_bool(
        data,
        lambda da: ~da.isin(test_elements),
    )


@register_builtin_check(
    error="str_matches('{pattern}')",
)
def str_matches(
    data: XrLike, pattern: str | re.Pattern
) -> bool | xr.DataArray:
    """Regex match at the start of each string (``re.match`` semantics)."""
    pat = pattern

    def pred(s: str) -> bool:
        if isinstance(pat, re.Pattern):
            return pat.match(s) is not None
        return re.match(pat, s) is not None

    if isinstance(data, xr.DataArray):
        return _str_mask_predicate(data, pred)
    return _dataset_reduce_bool(data, lambda da: _str_mask_predicate(da, pred))


@register_builtin_check(
    error="str_contains('{pattern}')",
)
def str_contains(
    data: XrLike, pattern: str | re.Pattern
) -> bool | xr.DataArray:
    """Regex search anywhere in each string (``re.search`` semantics)."""
    pat = pattern

    def pred(s: str) -> bool:
        if isinstance(pat, re.Pattern):
            return pat.search(s) is not None
        return re.search(pat, s) is not None

    if isinstance(data, xr.DataArray):
        return _str_mask_predicate(data, pred)
    return _dataset_reduce_bool(data, lambda da: _str_mask_predicate(da, pred))


@register_builtin_check(
    error="str_startswith('{string}')",
)
def str_startswith(data: XrLike, string: str) -> bool | xr.DataArray:
    """Each value's string form must start with ``string``."""
    if isinstance(data, xr.DataArray):
        return _str_mask_predicate(data, lambda s: s.startswith(string))
    return _dataset_reduce_bool(
        data,
        lambda da: _str_mask_predicate(da, lambda s: s.startswith(string)),
    )


@register_builtin_check(
    error="str_endswith('{string}')",
)
def str_endswith(data: XrLike, string: str) -> bool | xr.DataArray:
    """Each value's string form must end with ``string``."""
    if isinstance(data, xr.DataArray):
        return _str_mask_predicate(data, lambda s: s.endswith(string))
    return _dataset_reduce_bool(
        data,
        lambda da: _str_mask_predicate(da, lambda s: s.endswith(string)),
    )


@register_builtin_check(
    error="str_length",
)
def str_length(
    data: XrLike,
    min_value: int | None = None,
    max_value: int | None = None,
    exact_value: int | None = None,
) -> bool | xr.DataArray:
    """String length constraints per cell (``len(str(value))``).

    :raises ValueError: If ``exact_value`` and both bounds are omitted (same
        idea as the pandas builtin; the :class:`~pandera.api.checks.Check`
        factory usually catches this earlier).
    """
    if exact_value is None and min_value is None and max_value is None:
        raise ValueError(
            "Provide exact_value and/or at least one of min_value and max_value"
        )

    def _len_scalar(x: Any) -> int:
        # -1 marks unusable cells so comparisons like lens >= 2 still fail.
        if _str_element_unusable(x):
            return -1
        return len(str(x))

    def _mask_from_lens(lens: xr.DataArray) -> xr.DataArray:
        if exact_value is not None:
            return lens == exact_value
        if max_value is None:
            return lens >= min_value  # type: ignore[operator]
        if min_value is None:
            return lens <= max_value
        return (lens <= max_value) & (lens >= min_value)

    def _lengths_da(da: xr.DataArray) -> xr.DataArray:
        flat = np.asarray(da.values).ravel()
        lengths = np.asarray(
            np.frompyfunc(_len_scalar, 1, 1)(flat),
            dtype=int,
        )
        return xr.DataArray(
            lengths.reshape(np.asarray(da.values).shape),
            dims=da.dims,
            coords=da.coords,
        )

    if isinstance(data, xr.DataArray):
        return _mask_from_lens(_lengths_da(data))

    def _per(da: xr.DataArray) -> xr.DataArray:
        return _mask_from_lens(_lengths_da(da))

    return _dataset_reduce_bool(data, _per)


@register_builtin_check(
    error="unique_values_eq({values})",
)
def unique_values_eq(data: XrLike, values: Iterable) -> bool:
    """Set of unique values (union over Dataset vars) equals ``set(values)``.

    Uses :func:`numpy.unique` on raveled arrays (NaN behavior follows NumPy).
    """
    expected = set(values)

    def _uniq_set(da: xr.DataArray) -> set[Any]:
        flat = np.ravel(np.asarray(da.values))
        if flat.size == 0:
            return set()
        return set(np.unique(flat).tolist())

    if isinstance(data, xr.DataArray):
        return _uniq_set(data) == expected
    got: set[Any] = set()
    for _n, da in data.data_vars.items():
        got |= _uniq_set(da)
    return got == expected


@register_builtin_check(
    error="in_range({min_value}, {max_value})",
)
def in_range(
    data: XrLike,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
) -> bool | xr.DataArray:
    """Interval test on raw array values (per data var for Dataset).

    ``invalid='ignore'`` avoids floating-point warnings when values contain NaN;
    NaN cells still fail the interval test unless :class:`Check` uses
    ``ignore_na=True`` (handled in the xarray check backend).
    """
    left_op = operator.le if include_min else operator.lt
    right_op = operator.ge if include_max else operator.gt

    def _mask(arr: xr.DataArray) -> xr.DataArray:
        vals = np.asarray(arr.values)
        with np.errstate(invalid="ignore"):
            ok = left_op(min_value, vals) & right_op(max_value, vals)
        return xr.DataArray(ok, dims=arr.dims, coords=arr.coords)

    if isinstance(data, xr.DataArray):
        return _mask(data)
    for _name, da in data.data_vars.items():
        if not bool(np.all(_mask(da).values)):
            return False
    return True


@register_builtin_check(error="has_dims({dims})")
def has_dims(data: XrLike, dims: tuple[str, ...]) -> bool:
    """Every name in ``dims`` must appear on ``data.dims`` (subset check)."""
    return set(dims) <= set(data.dims)


@register_builtin_check(error="has_coords({coords})")
def has_coords(data: XrLike, coords: tuple[str, ...]) -> bool:
    """Each coordinate name must exist on ``data.coords``."""
    return all(name in data.coords for name in coords)


@register_builtin_check(error="has_attrs({attrs})")
def has_attrs(data: XrLike, attrs: dict[str, Any]) -> bool:
    """Global attrs must contain each key with an equal value (``==``)."""
    for key, val in attrs.items():
        if data.attrs.get(key) != val:
            return False
    return True


@register_builtin_check(error="ndim({n})")
def ndim(data: XrLike, n: int) -> bool:
    """DataArray: ``.ndim``; Dataset: number of dimension names."""
    if isinstance(data, xr.DataArray):
        return data.ndim == n
    return len(data.dims) == n


@register_builtin_check(error="dim_size({dim}, {size})")
def dim_size(data: XrLike, dim: str, size: int) -> bool:
    """``data.sizes[dim] == size`` (False if ``dim`` is missing)."""
    return dim in data.sizes and int(data.sizes[dim]) == int(size)


@register_builtin_check(error="is_monotonic({dim}, increasing={increasing})")
def is_monotonic(
    data: XrLike,
    dim: str,
    increasing: bool = True,
) -> bool:
    """1-D coordinate ``dim`` strictly increases or decreases (no ties)."""
    if dim not in data.coords:
        return False
    vals = np.asarray(data.coords[dim].values).ravel()
    if vals.size < 2:
        return True
    if increasing:
        return bool(np.all(vals[1:] > vals[:-1]))
    return bool(np.all(vals[1:] < vals[:-1]))


@register_builtin_check(error="no_duplicates_in_coord({coord})")
def no_duplicates_in_coord(data: XrLike, coord: str) -> bool:
    """Coordinate 1-D values are unique (after ravel)."""
    if coord not in data.coords:
        return False
    flat = np.asarray(data.coords[coord].values).ravel()
    if flat.size <= 1:
        return True
    return int(flat.size) == int(np.unique(flat).size)
