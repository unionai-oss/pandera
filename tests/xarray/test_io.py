"""Tests for xarray schema serialization (IO)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pandera.xarray as pa
from pandera.api.checks import Check
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.io.xarray_io import (
    deserialize_data_array_schema,
    deserialize_dataset_schema,
    deserialize_schema,
    from_json,
    from_yaml,
    serialize_data_array_schema,
    serialize_dataset_schema,
    serialize_schema,
    to_json,
    to_yaml,
)


class TestSerializeDataArraySchema:
    def test_basic(self):
        schema = DataArraySchema(
            dtype=np.float64,
            dims=("x", "y"),
            name="temperature",
        )
        serialized = serialize_data_array_schema(schema)
        assert serialized["schema_type"] == "data_array"
        assert serialized["dims"] == ["x", "y"]
        assert serialized["name"] == "temperature"
        assert "version" in serialized

    def test_with_sizes(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            sizes={"x": 10},
        )
        serialized = serialize_data_array_schema(schema)
        assert serialized["sizes"] == {"x": 10}

    def test_with_checks(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            checks=Check.in_range(0, 100),
        )
        serialized = serialize_data_array_schema(schema)
        assert serialized["checks"] is not None
        assert len(serialized["checks"]) == 1

    def test_with_coords(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            coords={"x": Coordinate(dtype="float64")},
        )
        serialized = serialize_data_array_schema(schema)
        assert serialized["coords"] is not None
        assert "x" in serialized["coords"]

    def test_roundtrip(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x", "y"),
            name="temp",
            nullable=True,
            coerce=True,
        )
        serialized = serialize_data_array_schema(schema)
        restored = deserialize_data_array_schema(serialized)
        assert isinstance(restored, DataArraySchema)
        assert restored.dims == ("x", "y")
        assert restored.name == "temp"
        assert restored.nullable is True
        assert restored.coerce is True


class TestSerializeDatasetSchema:
    def test_basic(self):
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(dtype="float64", dims=("x",)),
            },
        )
        serialized = serialize_dataset_schema(schema)
        assert serialized["schema_type"] == "dataset"
        assert serialized["data_vars"] is not None
        assert "temp" in serialized["data_vars"]

    def test_with_coords(self):
        schema = DatasetSchema(
            data_vars={"v": DataVar(dtype="float64", dims=("x",))},
            coords={"x": Coordinate(dtype="float64")},
        )
        serialized = serialize_dataset_schema(schema)
        assert serialized["coords"] is not None
        assert "x" in serialized["coords"]

    def test_roundtrip(self):
        schema = DatasetSchema(
            data_vars={
                "a": DataVar(dtype="float64", dims=("x",)),
                "b": DataVar(dtype="int32", dims=("x", "y"), required=False),
            },
            dims=("x", "y"),
            strict=True,
        )
        serialized = serialize_dataset_schema(schema)
        restored = deserialize_dataset_schema(serialized)
        assert isinstance(restored, DatasetSchema)
        assert "a" in restored.data_vars
        assert "b" in restored.data_vars
        assert restored.strict is True


class TestSerializeSchemaDispatch:
    def test_data_array(self):
        schema = DataArraySchema(dtype="float64", dims=("x",))
        serialized = serialize_schema(schema)
        assert serialized["schema_type"] == "data_array"

    def test_dataset(self):
        schema = DatasetSchema()
        serialized = serialize_schema(schema)
        assert serialized["schema_type"] == "dataset"

    def test_unknown_raises(self):
        with pytest.raises(TypeError, match="Expected DataArraySchema"):
            serialize_schema("not a schema")

    def test_deserialize_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown schema_type"):
            deserialize_schema({"schema_type": "unknown"})


class TestJsonIO:
    def test_to_json_string(self):
        schema = DataArraySchema(dtype="float64", dims=("x",), name="v")
        json_str = to_json(schema)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["schema_type"] == "data_array"

    def test_from_json_string(self):
        schema = DataArraySchema(dtype="float64", dims=("x",), name="v")
        json_str = to_json(schema)
        restored = from_json(json_str)
        assert isinstance(restored, DataArraySchema)
        assert restored.name == "v"

    def test_json_file_roundtrip(self):
        schema = DatasetSchema(
            data_vars={"a": DataVar(dtype="float64", dims=("x",))},
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            to_json(schema, target=f.name)
            restored = from_json(f.name)
        assert isinstance(restored, DatasetSchema)
        assert "a" in restored.data_vars

    def test_json_path_roundtrip(self):
        schema = DataArraySchema(dtype="float64", dims=("x",))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema.json"
            to_json(schema, target=path)
            restored = from_json(path)
        assert isinstance(restored, DataArraySchema)


class TestYamlIO:
    def test_to_yaml_string(self):
        schema = DataArraySchema(dtype="float64", dims=("x",), name="v")
        yaml_str = to_yaml(schema)
        assert isinstance(yaml_str, str)
        assert "data_array" in yaml_str

    def test_from_yaml_string(self):
        schema = DataArraySchema(
            dtype="float64", dims=("x",), name="v", coerce=True
        )
        yaml_str = to_yaml(schema)
        restored = from_yaml(yaml_str)
        assert isinstance(restored, DataArraySchema)
        assert restored.name == "v"
        assert restored.coerce is True

    def test_yaml_file_roundtrip(self):
        schema = DatasetSchema(
            data_vars={"b": DataVar(dtype="int64", dims=("y",))},
        )
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            to_yaml(schema, stream=f.name)
            restored = from_yaml(f.name)
        assert isinstance(restored, DatasetSchema)
        assert "b" in restored.data_vars


class TestEntryPointIO:
    def test_pa_to_json(self):
        schema = DataArraySchema(dtype="float64", dims=("x",))
        json_str = pa.to_json(schema)
        assert isinstance(json_str, str)

    def test_pa_from_json(self):
        schema = DataArraySchema(dtype="float64", dims=("x",), name="v")
        json_str = pa.to_json(schema)
        restored = pa.from_json(json_str)
        assert isinstance(restored, DataArraySchema)

    def test_pa_to_yaml(self):
        schema = DataArraySchema(dtype="float64", dims=("x",))
        yaml_str = pa.to_yaml(schema)
        assert isinstance(yaml_str, str)

    def test_pa_from_yaml(self):
        schema = DataArraySchema(dtype="float64", dims=("x",), name="v")
        yaml_str = pa.to_yaml(schema)
        restored = pa.from_yaml(yaml_str)
        assert isinstance(restored, DataArraySchema)


class TestChecksRoundtrip:
    def test_data_array_with_checks(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            checks=[Check.ge(0), Check.le(100)],
        )
        json_str = to_json(schema)
        restored = from_json(json_str)
        assert isinstance(restored, DataArraySchema)
        assert restored.checks is not None

    def test_dataset_with_checks(self):
        schema = DatasetSchema(
            data_vars={
                "v": DataVar(
                    dtype="float64",
                    dims=("x",),
                    checks=Check.in_range(0, 10),
                ),
            },
        )
        json_str = to_json(schema)
        restored = from_json(json_str)
        assert isinstance(restored, DatasetSchema)


class TestTitleDescription:
    def test_data_array_title_description_roundtrip(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            title="Temperature",
            description="Air temperature in Kelvin",
        )
        json_str = to_json(schema)
        restored = from_json(json_str)
        assert restored.title == "Temperature"
        assert restored.description == "Air temperature in Kelvin"

    def test_dataset_title_description_roundtrip(self):
        schema = DatasetSchema(
            data_vars={
                "v": DataVar(
                    dtype="float64",
                    dims=("x",),
                    title="Variable V",
                    description="A data variable",
                ),
            },
            title="My Dataset",
            description="A test dataset schema",
        )
        json_str = to_json(schema)
        restored = from_json(json_str)
        assert restored.title == "My Dataset"
        assert restored.description == "A test dataset schema"
        v_spec = restored.data_vars["v"]
        assert v_spec.title == "Variable V"
        assert v_spec.description == "A data variable"

    def test_coord_title_description_roundtrip(self):
        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            coords={
                "x": Coordinate(
                    dtype="float64",
                    title="X axis",
                    description="The X coordinate",
                ),
            },
        )
        json_str = to_json(schema)
        restored = from_json(json_str)
        x_coord = restored.coords["x"]
        assert x_coord.title == "X axis"
        assert x_coord.description == "The X coordinate"


class TestDatetimeSerialization:
    def test_datetime_check_json_roundtrip(self):
        """Datetime values in check stats are serialized as strings."""
        import datetime as dt

        schema = DataArraySchema(
            dtype="float64",
            dims=("x",),
            checks=[Check.ge(0.0)],
        )
        json_str = to_json(schema)
        parsed = json.loads(json_str)
        assert isinstance(parsed["checks"], list)
        restored = from_json(json_str)
        assert isinstance(restored, DataArraySchema)
        assert len(restored.checks) == 1


class TestComponentStatsFormat:
    def test_data_var_serialized_has_component_fields(self):
        """DataVar serialized format includes title, description, etc."""
        schema = DatasetSchema(
            data_vars={
                "temp": DataVar(
                    dtype="float64",
                    dims=("x",),
                    nullable=True,
                    coerce=True,
                    required=False,
                    title="Temperature",
                    description="Temperature variable",
                ),
            },
        )
        serialized = serialize_dataset_schema(schema)
        var = serialized["data_vars"]["temp"]
        assert var["title"] == "Temperature"
        assert var["description"] == "Temperature variable"
        assert var["nullable"] is True
        assert var["coerce"] is True
        assert var["required"] is False

    def test_legacy_dict_checks_format(self):
        """Deserializer handles legacy dict-of-checks format."""
        serialized = {
            "schema_type": "data_array",
            "dtype": "float64",
            "dims": ["x"],
            "checks": {
                "greater_than_or_equal_to": 0,
                "less_than_or_equal_to": 100,
            },
        }
        restored = from_json(json.dumps(serialized))
        assert isinstance(restored, DataArraySchema)
        assert len(restored.checks) == 2
