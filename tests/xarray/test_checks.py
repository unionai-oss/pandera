"""Xarray-specific built-in :class:`~pandera.api.checks.Check` tests."""

import re

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.api.xarray.components import DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.xarray import Check

# --- Numeric / comparison builtins (pandas parity) ---


def test_equal_to_and_not_equal_to():
    da_ok = xr.DataArray(np.array([1, 1, 1]), dims="x")
    DataArraySchema(checks=Check.equal_to(1)).validate(da_ok)
    da_mix = xr.DataArray(np.array([1, 1, 2]), dims="x")
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.equal_to(1)).validate(da_mix)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.equal_to(2)).validate(da_ok)
    DataArraySchema(checks=Check.not_equal_to(3)).validate(da_mix)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.not_equal_to(1)).validate(da_ok)


def test_comparison_aliases_instantiate():
    assert Check.eq(1).name == "equal_to"
    assert Check.gt(0).name == "greater_than"


def test_greater_less_chained():
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x")
    DataArraySchema(
        checks=[
            Check.greater_than(0),
            Check.less_than(4),
            Check.greater_than_or_equal_to(1),
            Check.less_than_or_equal_to(3),
        ],
    ).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.greater_than(2)).validate(da)


def test_less_than_none_raises_at_check_factory():
    with pytest.raises(ValueError, match="max_value must not be None"):
        Check.less_than(None)
    with pytest.raises(ValueError, match="max_value must not be None"):
        Check.less_than_or_equal_to(None)


def test_in_range_exclusive_bounds():
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x")
    DataArraySchema(
        checks=Check.in_range(1, 3, include_min=True, include_max=True),
    ).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(
            checks=Check.in_range(1, 3, include_min=False, include_max=True),
        ).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(
            checks=Check.in_range(1, 3, include_min=True, include_max=False),
        ).validate(da)


def test_isin_notin():
    da = xr.DataArray(np.array([1, 2, 3]), dims="x")
    DataArraySchema(checks=Check.isin([1, 2, 3])).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.isin([1, 2])).validate(da)
    DataArraySchema(checks=Check.notin([9, 10])).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.notin([1])).validate(da)


def test_isin_notin_after_frozenset_kwargs():
    """Check stores ``allowed_values`` as frozenset; xarray isin needs a list."""
    da = xr.DataArray(np.array([1, 2]), dims="x")
    c_in = Check.isin([1, 2])
    assert isinstance(c_in._check_kwargs["allowed_values"], frozenset)
    assert c_in(da).check_passed
    c_out = Check.notin([9])
    assert c_out(da).check_passed


def test_unique_values_eq():
    da = xr.DataArray(np.array([1, 2, 1, 2]), dims="x")
    DataArraySchema(checks=Check.unique_values_eq({1, 2})).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.unique_values_eq({1})).validate(da)


def test_unique_values_eq_empty_array():
    da = xr.DataArray(np.array([]), dims="x")
    DataArraySchema(checks=Check.unique_values_eq(set())).validate(da)


def test_unique_values_eq_dataset_union_across_vars():
    ds = xr.Dataset(
        {"a": ("x", np.array([1, 2])), "b": ("x", np.array([2, 3]))},
        coords={"x": np.arange(2)},
    )
    DatasetSchema(
        data_vars={"a": DataVar(), "b": DataVar()},
        checks=Check.unique_values_eq({1, 2, 3}),
    ).validate(ds)


# --- String builtins ---


def test_str_matches_contains_startswith_endswith():
    da = xr.DataArray(np.array(["abc", "ab_"]), dims="x")
    DataArraySchema(checks=Check.str_matches("ab.*")).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.str_matches("^z")).validate(da)

    DataArraySchema(checks=Check.str_contains("b")).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.str_contains("z")).validate(da)

    DataArraySchema(checks=Check.str_startswith("ab")).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.str_startswith("z")).validate(da)

    DataArraySchema(checks=Check.str_endswith("c")).validate(
        xr.DataArray(np.array(["abc", "c"]), dims="x")
    )
    da2 = xr.DataArray(np.array(["x.c", "y.c"]), dims="x")
    DataArraySchema(checks=Check.str_endswith(".c")).validate(da2)


def test_str_matches_compiled_pattern():
    da = xr.DataArray(np.array(["abc"]), dims="x")
    DataArraySchema(checks=Check.str_matches(re.compile("ab"))).validate(da)


def test_str_contains_compiled_pattern():
    da = xr.DataArray(np.array(["xyz"]), dims="x")
    DataArraySchema(checks=Check.str_contains(re.compile("y"))).validate(da)


def test_str_nan_and_none_fail_string_checks():
    da = xr.DataArray(np.array(["ok", np.nan], dtype=object), dims="x")
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.str_startswith("o")).validate(da)


def test_str_length_exact_and_range():
    da = xr.DataArray(np.array(["a", "bb", "ccc"]), dims="x")
    DataArraySchema(checks=Check.str_length(3)).validate(
        xr.DataArray(np.array(["xxx", "yyy"]), dims="x")
    )
    DataArraySchema(checks=Check.str_length(1, 3)).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.str_length(5, 5)).validate(da)


def test_str_length_factory_requires_bounds():
    with pytest.raises(ValueError, match="At least a minimum"):
        Check.str_length()


# --- Structural / xarray-specific ---


def test_has_dims():
    da = xr.DataArray(np.zeros((2, 3)), dims=("a", "b"))
    DataArraySchema(checks=Check.has_dims(("a", "b"))).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.has_dims(("a", "missing"))).validate(da)


def test_has_dims_empty_tuple_is_vacuous():
    da = xr.DataArray(np.zeros((2, 3)), dims=("a", "b"))
    DataArraySchema(checks=Check.has_dims(())).validate(da)


def test_builtin_str_length_raises_without_constraints():
    import pandera.backends.xarray.builtin_checks as xb

    da = xr.DataArray(np.array(["x"]), dims="x")
    with pytest.raises(ValueError, match="exact_value"):
        xb.str_length(da)


def test_has_coords():
    da = xr.DataArray(
        np.zeros(2),
        dims="x",
        coords={"x": ("x", [0, 1]), "meta": ("x", [3, 4])},
    )
    DataArraySchema(checks=Check.has_coords(("x", "meta"))).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.has_coords(("x", "missing"))).validate(da)


def test_has_attrs():
    da = xr.DataArray(np.zeros(2), dims="x", attrs={"units": "m"})
    DataArraySchema(checks=Check.has_attrs({"units": "m"})).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.has_attrs({"units": "s"})).validate(da)


def test_ndim_dataarray():
    da = xr.DataArray(np.zeros((2, 3)), dims=("a", "b"))
    DataArraySchema(checks=Check.ndim(2)).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.ndim(3)).validate(da)


def test_ndim_dataset():
    ds = xr.Dataset({"v": (["x", "y"], np.zeros((2, 3)))})
    DatasetSchema(
        data_vars={"v": DataVar()},
        checks=Check.ndim(2),
    ).validate(ds)
    with pytest.raises(pandera.errors.SchemaError):
        DatasetSchema(
            data_vars={"v": DataVar()},
            checks=Check.ndim(99),
        ).validate(ds)


def test_dim_size():
    da = xr.DataArray(np.zeros((2, 3)), dims=("a", "b"))
    DataArraySchema(checks=Check.dim_size("a", 2)).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.dim_size("a", 99)).validate(da)
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.dim_size("missing_dim", 1)).validate(da)


def test_is_monotonic():
    da = xr.DataArray(
        np.arange(3.0),
        dims="t",
        coords={"t": ("t", [0.0, 1.0, 2.0])},
    )
    DataArraySchema(checks=Check.is_monotonic("t")).validate(da)
    bad = da.assign_coords(t=("t", [2.0, 1.0, 0.0]))
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.is_monotonic("t")).validate(bad)


def test_is_monotonic_decreasing():
    da = xr.DataArray(
        np.arange(3.0),
        dims="t",
        coords={"t": ("t", [2.0, 1.0, 0.0])},
    )
    DataArraySchema(
        checks=Check.is_monotonic("t", increasing=False),
    ).validate(da)


def test_is_monotonic_missing_coord_fails():
    da = xr.DataArray(np.zeros(2), dims="x")
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.is_monotonic("t")).validate(da)


def test_is_monotonic_single_point_is_trivially_ok():
    da = xr.DataArray(
        [1.0],
        dims="t",
        coords={"t": ("t", [0.0])},
    )
    DataArraySchema(checks=Check.is_monotonic("t")).validate(da)


def test_is_monotonic_duplicate_breaks_strict():
    da = xr.DataArray(
        np.ones(3),
        dims="t",
        coords={"t": ("t", [0.0, 0.0, 1.0])},
    )
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(checks=Check.is_monotonic("t")).validate(da)


def test_no_duplicates_in_coord():
    da = xr.DataArray(
        np.zeros(3),
        dims="x",
        coords={"x": ("x", [0, 1, 2])},
    )
    DataArraySchema(
        checks=Check.no_duplicates_in_coord("x"),
    ).validate(da)
    dup = da.assign_coords(x=("x", [0, 0, 1]))
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(
            checks=Check.no_duplicates_in_coord("x"),
        ).validate(dup)


def test_no_duplicates_missing_coord_fails():
    da = xr.DataArray(np.zeros(2), dims="x")
    with pytest.raises(pandera.errors.SchemaError):
        DataArraySchema(
            checks=Check.no_duplicates_in_coord("missing"),
        ).validate(da)


def test_builtin_on_dataset():
    ds = xr.Dataset(
        {"a": (["x"], np.zeros(3))},
        coords={"x": ("x", [0, 1, 2])},
        attrs={"source": "test"},
    )
    DatasetSchema(
        data_vars={"a": DataVar()},
        checks=[
            Check.has_dims(("x",)),
            Check.has_coords(("x",)),
            Check.has_attrs({"source": "test"}),
            Check.no_duplicates_in_coord("x"),
        ],
    ).validate(ds)


def test_dataset_level_numeric_builtin_all_vars():
    ds = xr.Dataset(
        {
            "a": ("x", np.array([1, 2])),
            "b": ("x", np.array([3, 4])),
        },
        coords={"x": np.arange(2)},
    )
    DatasetSchema(
        data_vars={"a": DataVar(), "b": DataVar()},
        checks=[Check.greater_than(0), Check.less_than(10)],
    ).validate(ds)
    with pytest.raises(pandera.errors.SchemaError):
        DatasetSchema(
            data_vars={"a": DataVar(), "b": DataVar()},
            checks=Check.less_than(2),
        ).validate(ds)


def test_lazy_checks_collect_multiple_builtin_failures():
    da = xr.DataArray(np.array([0, 10]), dims="x")
    schema = DataArraySchema(
        checks=[
            Check.greater_than(0),
            Check.less_than(5),
        ],
    )
    with pytest.raises(pandera.errors.SchemaErrors) as exc:
        schema.validate(da, lazy=True)
    assert len(exc.value.schema_errors) >= 2
