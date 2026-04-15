"""Tests for CF convention checks on xarray objects."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.xarray import Check, DataArraySchema, DatasetSchema, DataVar


class TestCfStandardName:
    """Check.cf_standard_name — lightweight attrs check."""

    def test_pass(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="x",
            attrs={"standard_name": "air_temperature"},
        )
        schema = DataArraySchema(
            checks=Check.cf_standard_name("air_temperature"),
        )
        schema.validate(da)

    def test_fail_wrong_name(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="x",
            attrs={"standard_name": "wind_speed"},
        )
        schema = DataArraySchema(
            checks=Check.cf_standard_name("air_temperature"),
        )
        with pytest.raises(
            (
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            )
        ):
            schema.validate(da)

    def test_fail_missing_attr(self):
        da = xr.DataArray([1.0, 2.0], dims="x")
        schema = DataArraySchema(
            checks=Check.cf_standard_name("air_temperature"),
        )
        with pytest.raises(
            (
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            )
        ):
            schema.validate(da)

    def test_on_dataset(self):
        ds = xr.Dataset(
            {"temp": ("x", [1.0, 2.0])},
            attrs={"standard_name": "air_temperature"},
        )
        schema = DatasetSchema(
            data_vars={"temp": DataVar(dims=("x",))},
            checks=Check.cf_standard_name("air_temperature"),
        )
        schema.validate(ds)


class TestCfUnits:
    """Check.cf_units — lightweight attrs check."""

    def test_pass(self):
        da = xr.DataArray([1.0, 2.0], dims="x", attrs={"units": "K"})
        schema = DataArraySchema(
            checks=Check.cf_units("K"),
        )
        schema.validate(da)

    def test_fail_wrong(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="x",
            attrs={"units": "degC"},
        )
        schema = DataArraySchema(
            checks=Check.cf_units("K"),
        )
        with pytest.raises(
            (
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            )
        ):
            schema.validate(da)

    def test_fail_missing(self):
        da = xr.DataArray([1.0, 2.0], dims="x")
        schema = DataArraySchema(
            checks=Check.cf_units("K"),
        )
        with pytest.raises(
            (
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            )
        ):
            schema.validate(da)


class TestCfHasCellMethods:
    """Check.cf_has_cell_methods — lightweight attrs check."""

    def test_pass(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="x",
            attrs={"cell_methods": "time: mean"},
        )
        schema = DataArraySchema(
            checks=Check.cf_has_cell_methods("time: mean"),
        )
        schema.validate(da)

    def test_fail(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="x",
            attrs={"cell_methods": "time: max"},
        )
        schema = DataArraySchema(
            checks=Check.cf_has_cell_methods("time: mean"),
        )
        with pytest.raises(
            (
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            )
        ):
            schema.validate(da)


class TestCfHasStandardNames:
    """Check.cf_has_standard_names — requires cf_xarray."""

    def test_pass(self):
        cf_xarray = pytest.importorskip("cf_xarray")  # noqa: F841
        ds = xr.Dataset(
            {
                "temp": xr.DataArray(
                    [1.0, 2.0],
                    dims="x",
                    attrs={
                        "standard_name": "air_temperature",
                    },
                ),
            }
        )
        schema = DatasetSchema(
            data_vars={"temp": DataVar(dims=("x",))},
            checks=Check.cf_has_standard_names(["air_temperature"]),
        )
        schema.validate(ds)

    def test_fail_missing_standard_name(self):
        cf_xarray = pytest.importorskip("cf_xarray")  # noqa: F841
        ds = xr.Dataset({"temp": xr.DataArray([1.0], dims="x")})
        schema = DatasetSchema(
            data_vars={"temp": DataVar(dims=("x",))},
            checks=Check.cf_has_standard_names(["air_temperature"]),
        )
        with pytest.raises(
            (
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            )
        ):
            schema.validate(ds)

    def test_import_error_without_cf_xarray(self, monkeypatch):
        """If cf_xarray is not installed, raise an error."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cf_xarray":
                raise ImportError("no cf_xarray")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        da = xr.DataArray(
            [1.0],
            dims="x",
            attrs={"standard_name": "air_temperature"},
        )
        schema = DataArraySchema(
            checks=Check.cf_has_standard_names(["air_temperature"]),
        )
        with pytest.raises(
            (
                ImportError,
                pandera.errors.SchemaError,
                pandera.errors.SchemaErrors,
            ),
            match="cf_xarray",
        ):
            schema.validate(da)
