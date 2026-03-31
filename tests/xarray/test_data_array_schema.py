"""Tests for DataArraySchema."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.xarray import Check, Coordinate, DataArraySchema


def test_dtype_and_dims_pass():
    da = xr.DataArray(
        np.zeros((2, 3)),
        dims=("x", "y"),
        name="a",
    )
    schema = DataArraySchema(
        dtype=np.float64,
        dims=("x", "y"),
        name="a",
    )
    out = schema.validate(da)
    assert out.identical(da)


def test_dims_mismatch_raises():
    da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
    schema = DataArraySchema(dims=("x", "z"))
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_sizes():
    da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
    schema = DataArraySchema(sizes={"x": 2, "y": 3})
    schema.validate(da)
    bad = DataArraySchema(sizes={"x": 99})
    with pytest.raises(pandera.errors.SchemaError):
        bad.validate(da)


def test_coordinate_dimension_and_indexed():
    da = xr.DataArray(
        np.arange(3.0),
        dims="t",
        coords={
            "t": ("t", np.arange(3)),
            "aux": ("t", np.array([1.0, 2.0, 3.0])),
        },
    )
    schema = DataArraySchema(
        coords={
            "t": Coordinate(dimension=True, indexed=True),
            "aux": Coordinate(dimension=False, indexed=False),
        },
    )
    schema.validate(da)


def test_strict_coords():
    da = xr.DataArray(
        np.zeros((2,)),
        dims=("x",),
        coords={"x": ("x", np.arange(2))},
    )
    schema = DataArraySchema(coords=["x"], strict_coords=True)
    schema.validate(da)
    schema_bad = DataArraySchema(
        coords=["x"],
        strict_coords=True,
    )
    da_extra = da.assign_coords(extra=("x", np.zeros(2)))
    with pytest.raises(pandera.errors.SchemaError):
        schema_bad.validate(da_extra)


def test_data_check_and_lazy():
    da = xr.DataArray(np.array([1.0, 5.0, 3.0]), dims=("x",))
    schema = DataArraySchema(checks=Check(lambda x: x.max() < 4))
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(da, lazy=True)


def test_schema_only_skips_data_checks():
    da = xr.DataArray(np.array([1.0, 99.0]), dims=("x",))
    schema = DataArraySchema(checks=Check(lambda x: x.max() < 4))
    from pandera.config import ValidationDepth, config_context

    with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
        schema.validate(da)


def test_coerce_dtype():
    da = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
    schema = DataArraySchema(dtype=np.int64, coerce=True)
    out = schema.validate(da)
    assert np.issubdtype(out.dtype, np.integer)


def test_sizes_and_shape_mutually_exclusive():
    from pandera.errors import SchemaDefinitionError

    with pytest.raises(SchemaDefinitionError):
        DataArraySchema(sizes={"x": 1}, shape=(1,))


def test_nullable_rejects_nan():
    da = xr.DataArray(np.array([1.0, np.nan]), dims=("x",))
    schema = DataArraySchema(nullable=False)
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_nullable_true_allows_nan():
    da = xr.DataArray(np.array([1.0, np.nan]), dims=("x",))
    schema = DataArraySchema(nullable=True)
    schema.validate(da)


def test_in_range_ignore_na_skips_nan_cells():
    da = xr.DataArray(np.array([1.0, np.nan, 2.0]), dims=("x",))
    # nulls allowed at schema level; ignore_na only affects the in_range mask.
    schema = DataArraySchema(
        nullable=True,
        checks=Check.in_range(0, 3, ignore_na=True),
    )
    schema.validate(da)
    strict = DataArraySchema(
        nullable=True,
        checks=Check.in_range(0, 3, ignore_na=False),
    )
    with pytest.raises(pandera.errors.SchemaError):
        strict.validate(da)


def test_strict_attrs():
    da = xr.DataArray(np.zeros(2), dims="x", attrs={"a": 1, "b": 2})
    schema = DataArraySchema(attrs={"a": 1}, strict_attrs=True)
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_pydantic_attrs_pass():
    from pydantic import BaseModel

    class Attrs(BaseModel):
        units: str
        version: int

    da = xr.DataArray(
        np.zeros(2),
        dims="x",
        attrs={"units": "K", "version": 3},
    )
    schema = DataArraySchema(attrs=Attrs)
    schema.validate(da)


def test_pydantic_attrs_fail():
    from pydantic import BaseModel

    class Attrs(BaseModel):
        units: str
        version: int

    da = xr.DataArray(
        np.zeros(2),
        dims="x",
        attrs={"units": "K", "version": "bad"},
    )
    schema = DataArraySchema(attrs=Attrs)
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(da)


def test_pydantic_attrs_lazy_collects_all():
    from pydantic import BaseModel

    class Attrs(BaseModel):
        units: str
        version: int

    da = xr.DataArray(
        np.zeros(2),
        dims="x",
        attrs={"units": 999, "version": "bad"},
    )
    schema = DataArraySchema(attrs=Attrs)
    with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
        schema.validate(da, lazy=True)
    assert len(exc_info.value.schema_errors) >= 2


def test_head_subsample_runs_checks_on_subset():
    da = xr.DataArray(np.arange(10.0), dims=("x",))
    schema = DataArraySchema(
        checks=Check(lambda z: float(z.max()) < 5),
    )
    schema.validate(da, head=5)


def test_head_on_zero_dim_array_is_noop_for_subsample():
    da = xr.DataArray(np.float64(42.0))
    schema = DataArraySchema(checks=Check(lambda z: float(z) == 42.0))
    schema.validate(da, head=3)


def test_in_range_builtin():
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("x",))
    schema = DataArraySchema(
        checks=Check.in_range(0, 4),
    )
    schema.validate(da)


def test_lazy_schema_errors_multi_message():
    da = xr.DataArray(np.array([1.0, 5.0]), dims=("x",))
    schema = DataArraySchema(
        checks=[
            Check(lambda x: x.max() < 4),
            Check(lambda x: x.min() > 2),
        ],
    )
    with pytest.raises(pandera.errors.SchemaErrors) as excinfo:
        schema.validate(da, lazy=True)
    assert len(excinfo.value.schema_errors) >= 2


def test_parser_runs():
    from pandera.api.parsers import Parser

    da = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
    schema = DataArraySchema(parsers=Parser(lambda x: x * 2))
    out = schema.validate(da)
    assert float(out.max().item()) == 4.0


def test_chunked_data_checks_skipped_by_default():
    pytest.importorskip("dask.array")
    import dask.array as dda

    da = xr.DataArray(dda.ones(4, chunks=2), dims="x")
    schema = DataArraySchema(
        checks=Check(lambda x: False),
    )
    schema.validate(da)


def test_chunked_data_checks_when_validation_depth_includes_data():
    pytest.importorskip("dask.array")
    import dask.array as dda

    from pandera.config import ValidationDepth, config_context

    da = xr.DataArray(dda.ones(4, chunks=2), dims="x")
    schema = DataArraySchema(
        checks=Check(lambda x: False),
    )
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(da, lazy=True)


# -------------------------------------------------------------------
# Unit tests for individual @validate_scope check methods
# -------------------------------------------------------------------


class TestDataArrayCheckMethods:
    """Test individual check methods on the backend."""

    @pytest.fixture()
    def backend(self):
        from pandera.backends.xarray.container import (
            DataArraySchemaBackend,
        )

        return DataArraySchemaBackend()

    def test_check_name_pass(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", name="a")
        schema = DataArraySchema(name="a")
        result = backend.check_name(da, schema)
        assert result.passed

    def test_check_name_fail(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", name="b")
        schema = DataArraySchema(name="a")
        result = backend.check_name(da, schema)
        assert not result.passed
        assert "expected name" in result.message

    def test_check_name_none_passes(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", name="b")
        schema = DataArraySchema(name=None)
        result = backend.check_name(da, schema)
        assert result.passed

    def test_check_dims_pass(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(dims=("x", "y"))
        results = backend.check_dims(da, schema)
        assert all(r.passed for r in results) or len(results) == 0

    def test_check_dims_wrong_name(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(dims=("x", "z"))
        results = backend.check_dims(da, schema)
        assert any(not r.passed for r in results)

    def test_check_dims_wrong_length(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(dims=("x",))
        results = backend.check_dims(da, schema)
        assert any(not r.passed for r in results)
        assert "ndim/dims length" in results[0].message

    def test_check_sizes_pass(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(sizes={"x": 2, "y": 3})
        results = backend.check_sizes(da, schema)
        assert len(results) == 0

    def test_check_sizes_fail(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(sizes={"x": 99})
        results = backend.check_sizes(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_shape_pass(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(shape=(2, 3))
        results = backend.check_shape(da, schema)
        assert len(results) == 0

    def test_check_shape_fail(self, backend):
        da = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        schema = DataArraySchema(shape=(2, 99))
        results = backend.check_shape(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_dtype_pass(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(dtype=np.float64)
        result = backend.check_dtype(da, schema)
        assert result.passed

    def test_check_dtype_fail(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(dtype=np.int64)
        result = backend.check_dtype(da, schema)
        assert not result.passed
        assert "expected dtype" in result.message

    def test_check_chunked_eager_pass(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(chunked=False)
        result = backend.check_chunked(da, schema)
        assert result.passed

    def test_check_chunked_eager_fail(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(chunked=True)
        result = backend.check_chunked(da, schema)
        assert not result.passed

    def test_check_array_type_pass(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(array_type=np.ndarray)
        result = backend.check_array_type(da, schema)
        assert result.passed

    def test_check_array_type_fail(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(array_type=list)
        result = backend.check_array_type(da, schema)
        assert not result.passed

    def test_check_nullable_pass(self, backend):
        da = xr.DataArray(np.array([1.0, 2.0]), dims="x")
        schema = DataArraySchema(nullable=False)
        result = backend.check_nullable(da, schema)
        assert result.passed

    def test_check_nullable_fail(self, backend):
        da = xr.DataArray(np.array([1.0, np.nan]), dims="x")
        schema = DataArraySchema(nullable=False)
        result = backend.check_nullable(da, schema)
        assert not result.passed

    def test_check_attrs_pass(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", attrs={"a": 1})
        schema = DataArraySchema(attrs={"a": 1})
        results = backend.check_attrs(da, schema)
        assert len(results) == 0

    def test_check_attrs_fail(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", attrs={"a": 2})
        schema = DataArraySchema(attrs={"a": 1})
        results = backend.check_attrs(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_strict_attrs_pass(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", attrs={"a": 1})
        schema = DataArraySchema(attrs={"a": 1}, strict_attrs=True)
        results = backend.check_strict_attrs(da, schema)
        assert len(results) == 0

    def test_check_strict_attrs_fail(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"a": 1, "extra": 2},
        )
        schema = DataArraySchema(attrs={"a": 1}, strict_attrs=True)
        results = backend.check_strict_attrs(da, schema)
        assert len(results) == 1
        assert "unexpected attribute" in results[0].message

    def test_check_coords_pass(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            coords={"x": ("x", np.arange(2))},
        )
        schema = DataArraySchema(coords=["x"])
        results = backend.check_coords(da, schema)
        assert len(results) == 0

    def test_check_coords_missing(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(coords={"y": Coordinate()})
        results = backend.check_coords(da, schema)
        assert any(not r.passed for r in results)

    def test_check_strict_coords_pass(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            coords={"x": ("x", np.arange(2))},
        )
        schema = DataArraySchema(coords=["x"], strict_coords=True)
        results = backend.check_strict_coords(da, schema)
        assert len(results) == 0

    def test_check_strict_coords_fail(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            coords={
                "x": ("x", np.arange(2)),
                "extra": ("x", np.zeros(2)),
            },
        )
        schema = DataArraySchema(coords=["x"], strict_coords=True)
        results = backend.check_strict_coords(da, schema)
        assert any(not r.passed for r in results)

    def test_schema_scope_checks_skipped_with_data_only(self, backend):
        from pandera.config import (
            ValidationDepth,
            config_context,
        )

        da = xr.DataArray(np.zeros(2), dims="x", name="wrong")
        schema = DataArraySchema(name="correct")
        with config_context(validation_depth=ValidationDepth.DATA_ONLY):
            result = backend.check_name(da, schema)
        assert result.passed

    # --- ordered_dims tests ---

    def test_check_dims_ordered_pass(self, backend):
        da = xr.DataArray(
            np.zeros((2, 3)),
            dims=("x", "y"),
        )
        schema = DataArraySchema(dims=("x", "y"), ordered_dims=True)
        results = backend.check_dims(da, schema)
        assert len(results) == 0

    def test_check_dims_ordered_fail(self, backend):
        da = xr.DataArray(
            np.zeros((3, 2)),
            dims=("y", "x"),
        )
        schema = DataArraySchema(dims=("x", "y"), ordered_dims=True)
        results = backend.check_dims(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_dims_unordered_pass(self, backend):
        da = xr.DataArray(
            np.zeros((3, 2)),
            dims=("y", "x"),
        )
        schema = DataArraySchema(dims=("x", "y"), ordered_dims=False)
        results = backend.check_dims(da, schema)
        assert len(results) == 0

    def test_check_dims_unordered_fail(self, backend):
        da = xr.DataArray(
            np.zeros((3, 2)),
            dims=("y", "x"),
        )
        schema = DataArraySchema(dims=("x", "z"), ordered_dims=False)
        results = backend.check_dims(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    # --- attrs regex / callable tests ---

    def test_check_attrs_regex_pass(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"units": "K"},
        )
        schema = DataArraySchema(attrs={"units": "^(K|°C|°F)$"})
        results = backend.check_attrs(da, schema)
        assert len(results) == 0

    def test_check_attrs_regex_fail(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"units": "meters"},
        )
        schema = DataArraySchema(attrs={"units": "^(K|°C|°F)$"})
        results = backend.check_attrs(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_attrs_callable_pass(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"version": 3},
        )
        schema = DataArraySchema(
            attrs={"version": lambda v: isinstance(v, int) and v >= 2}
        )
        results = backend.check_attrs(da, schema)
        assert len(results) == 0

    def test_check_attrs_callable_fail(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"version": 1},
        )
        schema = DataArraySchema(
            attrs={"version": lambda v: isinstance(v, int) and v >= 2}
        )
        results = backend.check_attrs(da, schema)
        assert len(results) == 1
        assert not results[0].passed

    def test_check_attrs_missing_key(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x", attrs={})
        schema = DataArraySchema(attrs={"a": 1})
        results = backend.check_attrs(da, schema)
        assert len(results) == 1
        assert "missing" in results[0].message.lower()

    # --- Coordinate required tests ---

    def test_check_coords_optional_absent(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(coords={"label": Coordinate(required=False)})
        results = backend.check_coords(da, schema)
        assert len(results) == 0

    def test_check_coords_optional_present(self, backend):
        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            coords={
                "label": ("x", ["a", "b"]),
            },
        )
        schema = DataArraySchema(
            coords={"label": Coordinate(required=False, dtype=str)}
        )
        results = backend.check_coords(da, schema)
        assert len(results) == 0

    def test_check_coords_required_missing(self, backend):
        da = xr.DataArray(np.zeros(2), dims="x")
        schema = DataArraySchema(coords={"label": Coordinate(required=True)})
        results = backend.check_coords(da, schema)
        assert any(not r.passed for r in results)

    # --- pydantic BaseModel attrs tests ---

    def test_check_attrs_pydantic_pass(self, backend):
        from pydantic import BaseModel

        class Attrs(BaseModel):
            units: str
            version: int

        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"units": "K", "version": 3},
        )
        schema = DataArraySchema(attrs=Attrs)
        results = backend.check_attrs(da, schema)
        assert len(results) == 0

    def test_check_attrs_pydantic_wrong_type(self, backend):
        from pydantic import BaseModel

        class Attrs(BaseModel):
            units: str
            version: int

        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"units": "K", "version": "not_int"},
        )
        schema = DataArraySchema(attrs=Attrs)
        results = backend.check_attrs(da, schema)
        assert len(results) >= 1
        assert not results[0].passed
        assert "version" in results[0].message

    def test_check_attrs_pydantic_missing_field(self, backend):
        from pydantic import BaseModel

        class Attrs(BaseModel):
            units: str
            version: int

        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"units": "K"},
        )
        schema = DataArraySchema(attrs=Attrs)
        results = backend.check_attrs(da, schema)
        assert len(results) >= 1
        assert not results[0].passed
        assert "version" in results[0].message

    def test_check_attrs_pydantic_constrained(self, backend):
        from pydantic import BaseModel
        from pydantic import Field as PydanticField

        class Attrs(BaseModel):
            version: int = PydanticField(ge=2)

        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"version": 1},
        )
        schema = DataArraySchema(attrs=Attrs)
        results = backend.check_attrs(da, schema)
        assert len(results) >= 1
        assert not results[0].passed

    def test_check_strict_attrs_pydantic(self, backend):
        from pydantic import BaseModel

        class Attrs(BaseModel):
            units: str

        da = xr.DataArray(
            np.zeros(2),
            dims="x",
            attrs={"units": "K", "extra": 42},
        )
        schema = DataArraySchema(attrs=Attrs, strict_attrs=True)
        results = backend.check_strict_attrs(da, schema)
        assert len(results) == 1
        assert "extra" in results[0].message
