"""Stage 8: ``checks`` flow into xarray strategies.

The xarray strategy layer used to ignore ``checks`` arguments
(``# (unused) reserved for future check-aware generation``). After
Stage 8 every built-in check that ships with a constraint adapter
(numeric bounds, ``isin`` / ``notin``, equality, string regex /
length) routes through ``compile_dataarray_element_strategy`` and
constrains the generated values by construction (per
``specs/optimized-strategies.md`` §5).
"""

import numpy as np
import pytest
import xarray as xr

import pandera.backends.xarray.builtin_checks  # noqa: F401
from pandera.api.checks import Check
from pandera.api.xarray.components import DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema

pytest.importorskip("hypothesis")

import hypothesis
import hypothesis.strategies as hst

from pandera.strategies.xarray_strategies import (
    compile_dataarray_element_strategy,
    data_array_schema_strategy,
    data_array_strategy,
    dataset_schema_strategy,
    dataset_strategy,
)

# ----- compile_dataarray_element_strategy unit tests -----------------


def test_compile_with_bounds_emits_npst_from_dtype():
    """Numeric bounds short-circuit to ``npst.from_dtype`` kwargs."""
    from pandera.strategies.constraints import FieldConstraints

    constraints = FieldConstraints(min_value=0, max_value=100)
    strat = compile_dataarray_element_strategy(np.dtype("int64"), constraints)
    rendered = repr(strat)
    assert "filter" not in rendered


def test_compile_membership_uses_sampled_from():
    from pandera.strategies.constraints import FieldConstraints

    constraints = FieldConstraints(isin=frozenset({1, 2, 3, 4, 5}))
    strat = compile_dataarray_element_strategy(np.dtype("int64"), constraints)
    assert "sampled_from" in repr(strat)


def test_compile_equality_uses_just():
    from pandera.strategies.constraints import FieldConstraints

    constraints = FieldConstraints(eq=42)
    strat = compile_dataarray_element_strategy(np.dtype("int64"), constraints)
    assert "just" in repr(strat)


# ----- data_array_strategy with checks -------------------------------


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_data_array_strategy_honours_gt_check(data):
    da = data.draw(
        data_array_strategy(
            dtype="int64",
            dims=("x",),
            sizes={"x": 5},
            checks=[Check.gt(0)],
        )
    )
    assert isinstance(da, xr.DataArray)
    assert (da.values > 0).all()


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_data_array_strategy_honours_in_range(data):
    da = data.draw(
        data_array_strategy(
            dtype="int64",
            dims=("x",),
            sizes={"x": 5},
            checks=[Check.in_range(min_value=10, max_value=20)],
        )
    )
    assert (da.values >= 10).all()
    assert (da.values <= 20).all()


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_data_array_strategy_honours_isin(data):
    allowed = {1, 5, 9}
    da = data.draw(
        data_array_strategy(
            dtype="int64",
            dims=("x",),
            sizes={"x": 5},
            checks=[Check.isin(allowed)],
        )
    )
    assert set(da.values.tolist()) <= allowed


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_data_array_strategy_aggregates_multiple_bounds(data):
    """Stage 5/8: sibling bounds intersect into a single strategy."""
    da = data.draw(
        data_array_strategy(
            dtype="int64",
            dims=("x",),
            sizes={"x": 5},
            checks=[Check.gt(0), Check.lt(100), Check.notin([42])],
        )
    )
    vals = da.values.tolist()
    assert all(0 < v < 100 for v in vals)
    assert 42 not in vals


# ----- DataArraySchema strategy --------------------------------------


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_data_array_schema_strategy_honours_checks(data):
    schema = DataArraySchema(
        dtype=np.dtype("int64"),
        dims=("x",),
        checks=[Check.gt(0), Check.lt(50)],
    )
    da = data.draw(data_array_schema_strategy(schema, size=5))
    schema.validate(da)
    assert (da.values > 0).all()
    assert (da.values < 50).all()


# ----- dataset_strategy with per-DataVar checks ----------------------


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_dataset_strategy_honours_per_var_checks(data):
    ds = data.draw(
        dataset_strategy(
            data_vars={
                "temp": {
                    "dtype": "float64",
                    "dims": ("x",),
                    "checks": [Check.gt(0.0), Check.lt(100.0)],
                },
                "ids": {
                    "dtype": "int64",
                    "dims": ("x",),
                    "checks": [Check.in_range(1, 10)],
                },
            },
            sizes={"x": 5},
        )
    )
    assert (ds["temp"].values > 0).all()
    assert (ds["temp"].values < 100).all()
    assert (ds["ids"].values >= 1).all()
    assert (ds["ids"].values <= 10).all()


@hypothesis.given(hst.data())
@hypothesis.settings(max_examples=10)
def test_dataset_schema_strategy_honours_per_var_checks(data):
    schema = DatasetSchema(
        data_vars={
            "temp": DataVar(
                dtype=np.dtype("float64"),
                dims=("x",),
                checks=[Check.ge(0.0), Check.le(1.0)],
            )
        }
    )
    ds = data.draw(dataset_schema_strategy(schema, size=5))
    schema.validate(ds)
    assert (ds["temp"].values >= 0).all()
    assert (ds["temp"].values <= 1).all()
