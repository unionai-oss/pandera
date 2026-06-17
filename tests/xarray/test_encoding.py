"""Tests for encoding validation on xarray DataArrays and Datasets."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pandera.errors
from pandera.xarray import (
    Check,
    DataArraySchema,
    DatasetSchema,
    DataVar,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _da_with_encoding(encoding: dict):
    da = xr.DataArray(
        np.arange(6, dtype="float64"),
        dims="x",
    )
    da.encoding = encoding
    return da


def _ds_with_encoding(encoding: dict):
    ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
    ds.encoding = encoding
    return ds


# ------------------------------------------------------------------
# DataArraySchema — schema-level encoding
# ------------------------------------------------------------------


class TestDataArraySchemaEncoding:
    def test_pass_equality(self):
        da = _da_with_encoding({"_FillValue": -999.0, "dtype": "float32"})
        schema = DataArraySchema(
            encoding={"_FillValue": -999.0, "dtype": "float32"},
        )
        result = schema.validate(da)
        assert result is not None

    def test_fail_missing_key(self):
        da = _da_with_encoding({"dtype": "float32"})
        schema = DataArraySchema(
            encoding={"_FillValue": -999.0},
        )
        with pytest.raises(pandera.errors.SchemaErrors, match="missing"):
            schema.validate(da, lazy=True)

    def test_fail_wrong_value(self):
        da = _da_with_encoding({"_FillValue": 0.0})
        schema = DataArraySchema(
            encoding={"_FillValue": -999.0},
        )
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(da, lazy=True)

    def test_no_encoding_requirement_passes(self):
        da = _da_with_encoding({"anything": 42})
        schema = DataArraySchema()
        schema.validate(da)

    def test_empty_encoding_on_data_passes(self):
        da = _da_with_encoding({})
        schema = DataArraySchema()
        schema.validate(da)


# ------------------------------------------------------------------
# DatasetSchema — schema-level encoding
# ------------------------------------------------------------------


class TestDatasetSchemaEncoding:
    """Dataset-level encoding (ds.encoding): unlimited_dims, source."""

    def test_pass_equality(self):
        ds = _ds_with_encoding({"unlimited_dims": ["x"]})
        schema = DatasetSchema(
            encoding={"unlimited_dims": ["x"]},
        )
        result = schema.validate(ds)
        assert result is not None

    def test_fail_missing_key(self):
        ds = _ds_with_encoding({})
        schema = DatasetSchema(
            encoding={"unlimited_dims": ["time"]},
        )
        with pytest.raises(pandera.errors.SchemaErrors, match="missing"):
            schema.validate(ds, lazy=True)

    def test_fail_wrong_value(self):
        ds = _ds_with_encoding({"unlimited_dims": ["time"]})
        schema = DatasetSchema(
            encoding={"unlimited_dims": ["x"]},
        )
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(ds, lazy=True)


# ------------------------------------------------------------------
# Regex matching on encoding values
# ------------------------------------------------------------------


class TestEncodingRegex:
    def test_regex_match(self):
        da = _da_with_encoding({"dtype": "float32"})
        schema = DataArraySchema(
            encoding={"dtype": "^float.*"},
        )
        schema.validate(da)

    def test_regex_no_match(self):
        da = _da_with_encoding({"dtype": "int32"})
        schema = DataArraySchema(
            encoding={"dtype": "^float.*"},
        )
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(da, lazy=True)


# ------------------------------------------------------------------
# Callable matching on encoding values
# ------------------------------------------------------------------


class TestEncodingCallable:
    def test_callable_pass(self):
        da = _da_with_encoding({"scale_factor": 0.01})
        schema = DataArraySchema(
            encoding={
                "scale_factor": lambda v: 0 < v < 1,
            },
        )
        schema.validate(da)

    def test_callable_fail(self):
        da = _da_with_encoding({"scale_factor": 5.0})
        schema = DataArraySchema(
            encoding={
                "scale_factor": lambda v: 0 < v < 1,
            },
        )
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(da, lazy=True)


# ------------------------------------------------------------------
# Check.has_encoding() builtin check
# ------------------------------------------------------------------


class TestHasEncodingCheck:
    def test_dataarray_pass(self):
        da = _da_with_encoding({"_FillValue": -999.0, "dtype": "float32"})
        schema = DataArraySchema(
            checks=Check.has_encoding({"_FillValue": -999.0}),
        )
        schema.validate(da)

    def test_dataarray_fail(self):
        da = _da_with_encoding({"dtype": "float32"})
        schema = DataArraySchema(
            checks=Check.has_encoding({"_FillValue": -999.0}),
        )
        with pytest.raises(pandera.errors.SchemaError):
            schema.validate(da)

    def test_dataset_pass(self):
        ds = _ds_with_encoding({"compression": "zlib"})
        schema = DatasetSchema(
            checks=Check.has_encoding({"compression": "zlib"}),
        )
        schema.validate(ds)

    def test_dataset_fail(self):
        ds = _ds_with_encoding({})
        schema = DatasetSchema(
            checks=Check.has_encoding({"compression": "zlib"}),
        )
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(ds, lazy=True)


# ------------------------------------------------------------------
# Per-variable encoding via DataVar
# ------------------------------------------------------------------


class TestDataVarEncoding:
    """DataVar(encoding=...) validates ds[var].encoding."""

    def test_per_var_encoding_pass(self):
        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds["temp"].encoding = {
            "_FillValue": -999.0,
            "dtype": "float32",
        }
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding={
                        "_FillValue": -999.0,
                        "dtype": "float32",
                    },
                ),
            },
        )
        schema.validate(ds)

    def test_per_var_encoding_fail_wrong_value(self):
        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds["temp"].encoding = {"_FillValue": 0.0}
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding={"_FillValue": -999.0},
                ),
            },
        )
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(ds, lazy=True)

    def test_per_var_encoding_fail_missing_key(self):
        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds["temp"].encoding = {}
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding={"scale_factor": 0.1},
                ),
            },
        )
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(ds, lazy=True)

    def test_per_var_encoding_callable(self):
        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds["temp"].encoding = {"complevel": 4}
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding={
                        "complevel": lambda v: 1 <= v <= 9,
                    },
                ),
            },
        )
        schema.validate(ds)

    def test_per_var_and_dataset_encoding_together(self):
        """Per-variable and dataset-level encoding coexist."""
        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds.encoding = {"unlimited_dims": ["x"]}
        ds["temp"].encoding = {"_FillValue": -999.0}
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding={"_FillValue": -999.0},
                ),
            },
            encoding={"unlimited_dims": ["x"]},
        )
        schema.validate(ds)


# ------------------------------------------------------------------
# Pydantic BaseModel encoding validation
# ------------------------------------------------------------------


class TestPydanticEncoding:
    """Validate encoding via a pydantic BaseModel class."""

    def test_dataarray_pydantic_pass(self):
        from pydantic import BaseModel

        class Enc(BaseModel):
            scale_factor: float
            dtype: str

        da = _da_with_encoding({"scale_factor": 0.01, "dtype": "float32"})
        schema = DataArraySchema(encoding=Enc)
        schema.validate(da)

    def test_dataarray_pydantic_fail_missing_field(self):
        from pydantic import BaseModel

        class Enc(BaseModel):
            scale_factor: float
            dtype: str

        da = _da_with_encoding({"dtype": "float32"})
        schema = DataArraySchema(encoding=Enc)
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(da, lazy=True)

    def test_dataarray_pydantic_fail_wrong_type(self):
        from pydantic import BaseModel

        class Enc(BaseModel):
            scale_factor: float

        da = _da_with_encoding({"scale_factor": "not_a_number"})
        schema = DataArraySchema(encoding=Enc)
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(da, lazy=True)

    def test_dataset_pydantic_pass(self):
        from pydantic import BaseModel

        class DsEnc(BaseModel):
            unlimited_dims: list[str]

        ds = _ds_with_encoding({"unlimited_dims": ["x"]})
        schema = DatasetSchema(encoding=DsEnc)
        schema.validate(ds)

    def test_dataset_pydantic_fail(self):
        from pydantic import BaseModel

        class DsEnc(BaseModel):
            unlimited_dims: list[str]

        ds = _ds_with_encoding({"unlimited_dims": "not_a_list"})
        schema = DatasetSchema(encoding=DsEnc)
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema.validate(ds, lazy=True)

    def test_datavar_pydantic_encoding_pass(self):
        from pydantic import BaseModel

        class VarEnc(BaseModel):
            scale_factor: float
            dtype: str

        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds["temp"].encoding = {
            "scale_factor": 0.01,
            "dtype": "float32",
        }
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding=VarEnc,
                ),
            },
        )
        schema.validate(ds)

    def test_datavar_pydantic_encoding_fail(self):
        from pydantic import BaseModel

        class VarEnc(BaseModel):
            scale_factor: float

        ds = xr.Dataset({"temp": xr.DataArray(np.arange(4.0), dims="x")})
        ds["temp"].encoding = {"scale_factor": "bad"}
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dims=("x",),
                    encoding=VarEnc,
                ),
            },
        )
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(ds, lazy=True)

    def test_pydantic_encoding_with_validators(self):
        """Pydantic field_validator constraints are respected."""
        from pydantic import BaseModel, field_validator

        class StrictEnc(BaseModel):
            complevel: int

            @field_validator("complevel")
            @classmethod
            def check_range(cls, v: int) -> int:
                if not 1 <= v <= 9:
                    raise ValueError("complevel must be between 1 and 9")
                return v

        da = _da_with_encoding({"complevel": 4})
        schema = DataArraySchema(encoding=StrictEnc)
        schema.validate(da)

        da_bad = _da_with_encoding({"complevel": 99})
        schema_bad = DataArraySchema(encoding=StrictEnc)
        with pytest.raises(pandera.errors.SchemaErrors, match="encoding"):
            schema_bad.validate(da_bad, lazy=True)


# ------------------------------------------------------------------
# Schema-level encoding is SCHEMA scope
# ------------------------------------------------------------------


class TestEncodingValidationScope:
    def test_schema_only_still_checks_encoding(self):
        """Encoding is a SCHEMA-level check so it should run
        even under SCHEMA_ONLY depth."""
        from pandera.config import ValidationDepth, config_context

        da = _da_with_encoding({"_FillValue": 0.0})
        schema = DataArraySchema(
            encoding={"_FillValue": -999.0},
        )
        with config_context(
            validation_depth=ValidationDepth.SCHEMA_ONLY,
        ):
            with pytest.raises(
                pandera.errors.SchemaErrors,
                match="encoding",
            ):
                schema.validate(da, lazy=True)
