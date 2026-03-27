"""Tests for xarray decorator integration (check_input, check_output,
check_io, check_types)."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors  # noqa: E402
import pandera.xarray as pa  # noqa: E402
from pandera.typing.xarray import Coordinate, DataArray, Dataset  # noqa: E402

from tests.xarray.conftest import GridModel, SurfaceModel  # noqa: E402


# ===================================================================
# check_input / check_output with imperative schemas
# ===================================================================


class TestCheckInputOutput:
    """Tests for check_input and check_output with xarray schemas."""

    def test_check_input_data_array(self, grid_da):
        schema = pa.DataArraySchema(
            dtype=np.float64, dims=("x", "y"), name="values"
        )

        @pa.check_input(schema, "da")
        def process(da):
            return da * 2

        out = process(grid_da)
        assert float(out.max()) == 2.0

    def test_check_input_first_arg(self, grid_da):
        schema = pa.DataArraySchema(
            dtype=np.float64, dims=("x", "y"), name="values"
        )

        @pa.check_input(schema)
        def process(da):
            return da

        out = process(grid_da)
        assert out.identical(grid_da)

    def test_check_input_error_on_bad_data(self):
        schema = pa.DataArraySchema(dims=("x", "y"))
        bad_da = xr.DataArray(np.zeros(3), dims=("z",))

        @pa.check_input(schema, "da")
        def process(da):
            return da

        with pytest.raises(pandera.errors.SchemaError):
            process(bad_da)

    def test_check_output_data_array(self, grid_da):
        schema = pa.DataArraySchema(
            dtype=np.float64, dims=("x", "y"), name="values"
        )

        @pa.check_output(schema)
        def generate():
            return grid_da

        out = generate()
        assert out.identical(grid_da)

    def test_check_output_error_on_bad_data(self):
        schema = pa.DataArraySchema(name="expected_name")

        @pa.check_output(schema)
        def generate():
            return xr.DataArray(np.ones(3), dims="x", name="wrong")

        with pytest.raises(pandera.errors.SchemaError):
            generate()

    def test_check_input_dataset(self, surface_ds):
        schema = pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataVar(
                    dtype=np.float64, dims=("x",)
                ),
            },
        )

        @pa.check_input(schema, "ds")
        def process(ds):
            return ds

        out = process(surface_ds)
        assert out.identical(surface_ds)

    def test_check_output_dataset(self, surface_ds):
        schema = pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataVar(
                    dtype=np.float64, dims=("x",)
                ),
            },
        )

        @pa.check_output(schema)
        def generate():
            return surface_ds

        out = generate()
        assert out.identical(surface_ds)

    def test_check_input_by_index(self, grid_da):
        schema = pa.DataArraySchema(
            dtype=np.float64, dims=("x", "y"), name="values"
        )

        @pa.check_input(schema, 1)
        def process(unused, da):
            return da

        out = process("ignored", grid_da)
        assert out.identical(grid_da)

    def test_check_input_as_kwarg(self, grid_da):
        schema = pa.DataArraySchema(
            dtype=np.float64, dims=("x", "y"), name="values"
        )

        @pa.check_input(schema, "da")
        def process(da):
            return da

        out = process(da=grid_da)
        assert out.identical(grid_da)


# ===================================================================
# check_io with xarray schemas
# ===================================================================


class TestCheckIO:
    """Tests for check_io with xarray schemas."""

    def test_check_io_data_array(self, grid_da):
        schema = pa.DataArraySchema(
            dtype=np.float64, dims=("x", "y"), name="values"
        )

        @pa.check_io(da=schema, out=schema)
        def process(da):
            return da

        out = process(grid_da)
        assert out.identical(grid_da)

    def test_check_io_dataset(self, surface_ds):
        schema = pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataVar(
                    dtype=np.float64, dims=("x",)
                ),
            },
        )

        @pa.check_io(ds=schema, out=schema)
        def process(ds):
            return ds

        out = process(surface_ds)
        assert out.identical(surface_ds)

    def test_check_io_error_on_bad_input(self):
        schema = pa.DataArraySchema(dims=("x", "y"))
        bad = xr.DataArray(np.zeros(3), dims=("z",))

        @pa.check_io(da=schema)
        def process(da):
            return da

        with pytest.raises(pandera.errors.SchemaError):
            process(bad)

    def test_check_io_error_on_bad_output(self, grid_da):
        in_schema = pa.DataArraySchema(dims=("x", "y"))
        out_schema = pa.DataArraySchema(name="expected_name")

        @pa.check_io(da=in_schema, out=out_schema)
        def process(da):
            return da

        with pytest.raises(pandera.errors.SchemaError):
            process(grid_da)


# ===================================================================
# check_types with DataArrayModel
# ===================================================================


class TestCheckTypesDataArray:
    """Tests for check_types with DataArray[DataArrayModel]."""

    def test_basic(self, grid_da):
        @pa.check_types
        def process(
            da: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return da

        out = process(grid_da)
        assert out.identical(grid_da)

    def test_validates_input(self):
        bad = xr.DataArray(np.ones(3), dims=("z",), name="wrong")

        @pa.check_types
        def process(
            da: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return da

        with pytest.raises(pandera.errors.SchemaError):
            process(bad)

    def test_validates_output(self, grid_da):
        @pa.check_types
        def process(
            da: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return xr.DataArray(
                np.ones(3), dims=("z",), name="bad"
            )

        with pytest.raises(pandera.errors.SchemaError):
            process(grid_da)

    def test_multiple_args(self, grid_da):
        @pa.check_types
        def process(
            a: DataArray[GridModel],
            b: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return a

        out = process(grid_da, grid_da)
        assert out.identical(grid_da)

    def test_non_annotated_args_pass_through(self, grid_da):
        @pa.check_types
        def process(
            da: DataArray[GridModel],
            factor: int,
        ) -> DataArray[GridModel]:
            return da * factor

        out = process(grid_da, 1)
        assert out.identical(grid_da)

    def test_kwarg_input(self, grid_da):
        @pa.check_types
        def process(
            da: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return da

        out = process(da=grid_da)
        assert out.identical(grid_da)

    def test_with_parentheses(self, grid_da):
        @pa.check_types()
        def process(
            da: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return da

        out = process(grid_da)
        assert out.identical(grid_da)

    def test_lazy_kwarg(self, grid_da):
        @pa.check_types(lazy=True)
        def process(
            da: DataArray[GridModel],
        ) -> DataArray[GridModel]:
            return xr.DataArray(
                np.ones(3), dims=("z",), name="bad"
            )

        with pytest.raises(pandera.errors.SchemaErrors):
            process(grid_da)


# ===================================================================
# check_types with DatasetModel
# ===================================================================


class TestCheckTypesDataset:
    """Tests for check_types with Dataset[DatasetModel]."""

    def test_basic(self, surface_ds):
        @pa.check_types
        def process(
            ds: Dataset[SurfaceModel],
        ) -> Dataset[SurfaceModel]:
            return ds

        out = process(surface_ds)
        assert out.identical(surface_ds)

    def test_validates_input(self):
        bad = xr.Dataset({"wrong_var": (("z",), np.ones(3))})

        @pa.check_types
        def process(
            ds: Dataset[SurfaceModel],
        ) -> Dataset[SurfaceModel]:
            return ds

        with pytest.raises(pandera.errors.SchemaError):
            process(bad)

    def test_validates_output(self, surface_ds):
        @pa.check_types
        def process(
            ds: Dataset[SurfaceModel],
        ) -> Dataset[SurfaceModel]:
            return xr.Dataset(
                {"wrong_var": (("z",), np.ones(3))}
            )

        with pytest.raises(pandera.errors.SchemaError):
            process(surface_ds)

    def test_multiple_args(self, surface_ds):
        @pa.check_types
        def process(
            a: Dataset[SurfaceModel],
            b: Dataset[SurfaceModel],
        ) -> Dataset[SurfaceModel]:
            return a

        out = process(surface_ds, surface_ds)
        assert out.identical(surface_ds)

    def test_non_annotated_args_pass_through(self, surface_ds):
        @pa.check_types
        def process(
            ds: Dataset[SurfaceModel],
            label: str,
        ) -> Dataset[SurfaceModel]:
            return ds

        out = process(surface_ds, "hello")
        assert out.identical(surface_ds)


# ===================================================================
# check_types with mixed DataArray and Dataset annotations
# ===================================================================


class TestCheckTypesMixed:
    """check_types with both DataArray and Dataset annotations."""

    def test_data_array_input_dataset_output(self, grid_da):
        class OutputDS(pa.DatasetModel):
            values: np.float64 = pa.Field(dims=("x", "y"))
            x: Coordinate[np.float64]
            y: Coordinate[np.float64]

        @pa.check_types
        def process(
            da: DataArray[GridModel],
        ) -> Dataset[OutputDS]:
            return da.to_dataset()

        out = process(grid_da)
        assert isinstance(out, xr.Dataset)

    def test_unannotated_return_is_not_validated(self, grid_da):
        @pa.check_types
        def process(da: DataArray[GridModel]):
            return "not_a_data_array"

        assert process(grid_da) == "not_a_data_array"


# ===================================================================
# check_types with optional annotations
# ===================================================================


class TestCheckTypesOptional:
    """check_types with Optional xarray annotations."""

    def test_optional_input_none(self):
        from typing import Optional

        @pa.check_types
        def process(
            da: Optional[DataArray[GridModel]],
        ) -> int:
            return 42

        assert process(None) == 42

    def test_optional_input_valid(self, grid_da):
        from typing import Optional

        @pa.check_types
        def process(
            da: Optional[DataArray[GridModel]],
        ) -> DataArray[GridModel]:
            return da

        out = process(grid_da)
        assert out.identical(grid_da)
