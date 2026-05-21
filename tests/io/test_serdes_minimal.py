"""Serialization / deserialization tests, including ``minimal=True`` output."""

from __future__ import annotations

import json
from types import ModuleType

import pytest
import yaml

import pandera.pandas as pa
from pandera.io import pandas_io


class TestPandasSerdesMinimal:
    """Pandas :mod:`pandera.io.pandas_io` minimal vs full serialization."""

    def _simple_schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            {
                "x": pa.Column(int),
                "y": pa.Column(str, nullable=True),
            },
            strict=True,
            coerce=True,
        )

    def test_serialize_minimal_omits_version(self) -> None:
        schema = pa.DataFrameSchema({"a": pa.Column(int)})
        full = pandas_io.serialize_schema(schema, minimal=False)
        mini = pandas_io.serialize_schema(schema, minimal=True)
        assert "version" in full
        assert "version" not in mini
        assert full["schema_type"] == mini["schema_type"] == "dataframe"

    def test_serialize_minimal_omits_default_column_fields(self) -> None:
        schema = pa.DataFrameSchema({"a": pa.Column(int, nullable=False)})
        mini = pandas_io.serialize_schema(schema, minimal=True)
        col = mini["columns"]["a"]
        assert "nullable" not in col

    def test_serialize_full_includes_nullable_false(self) -> None:
        schema = pa.DataFrameSchema({"a": pa.Column(int, nullable=False)})
        full = pandas_io.serialize_schema(schema, minimal=False)
        assert full["columns"]["a"]["nullable"] is False

    def test_yaml_json_roundtrip_minimal_equals_original(self) -> None:
        schema = self._simple_schema()
        for dump, load in (
            (lambda s: pandas_io.to_yaml(s), pandas_io.from_yaml),
            (lambda s: pandas_io.to_json(s), pandas_io.from_json),
        ):
            payload = dump(schema)
            restored = load(payload)
            assert restored == schema

    def test_yaml_json_roundtrip_full_equals_original(self) -> None:
        schema = self._simple_schema()
        y = pandas_io.to_yaml(schema, minimal=False)
        assert yaml.safe_load(y)["version"] is not None
        assert pandas_io.from_yaml(y) == schema
        j = pandas_io.to_json(schema, minimal=False)
        assert json.loads(j)["version"] is not None
        assert pandas_io.from_json(j) == schema

    def test_minimal_yaml_shorter_than_full(self) -> None:
        schema = pa.DataFrameSchema(
            {
                "a": pa.Column(int, checks=pa.Check.ge(0)),
                "b": pa.Column(str),
            },
            strict=True,
        )
        short = pandas_io.to_yaml(schema, minimal=True)
        long = pandas_io.to_yaml(schema, minimal=False)
        assert len(short) < len(long)

    def test_minimal_preserves_dataframe_library_tag(self) -> None:
        schema = pa.DataFrameSchema({"a": pa.Column(int)})
        mini = pandas_io.serialize_schema(
            schema, dataframe_library="dask", minimal=True
        )
        assert mini.get("dataframe_library") == "dask"

    def test_schema_method_accepts_minimal(self) -> None:
        schema = pa.DataFrameSchema({"a": pa.Column(int)})
        y_min = schema.to_yaml(minimal=True)
        y_full = schema.to_yaml(minimal=False)
        assert y_min is not None
        assert y_full is not None
        assert "version" not in yaml.safe_load(y_min)
        assert "version" in yaml.safe_load(y_full)

    def test_minimal_strips_default_check_options(self) -> None:
        schema = pa.DataFrameSchema(
            {"a": pa.Column(int, checks=pa.Check.ge(0))},
        )
        y = pandas_io.to_yaml(schema, minimal=True)
        assert "ignore_na" not in y
        assert "raise_warning" not in y
        assert pandas_io.from_yaml(y) == schema

    def test_flat_check_keys_in_serialized_dict(self) -> None:
        """Checks serialize as Field-style keys, not a ``checks:`` list."""
        schema = pa.DataFrameSchema(
            {"a": pa.Column(int, checks=[pa.Check.ge(1), pa.Check.le(10)])},
        )
        d = pandas_io.serialize_schema(schema, minimal=True)
        col = d["columns"]["a"]
        assert "checks" not in col
        assert col["greater_than_or_equal_to"] == 1
        assert col["less_than_or_equal_to"] == 10

    def test_legacy_checks_list_yaml_still_loads(self) -> None:
        """YAML using the previous ``checks:`` list shape still deserializes."""
        legacy = """
schema_type: dataframe
columns:
  a:
    dtype: int64
    nullable: false
    checks:
    - value: 1
      options:
        check_name: greater_than_or_equal_to
    - value: 10
      options:
        check_name: less_than_or_equal_to
coerce: false
strict: false
"""
        loaded = pandas_io.from_yaml(legacy)
        checks = loaded.columns["a"].checks
        assert checks is not None and len(checks) == 2


class TestPolarsSerdesMinimal:
    """Polars IO roundtrip and minimal key omission."""

    def test_roundtrip_equality_yaml_and_json(self) -> None:
        import pandera.polars as pl_pa
        from pandera.io import polars_io

        schema = pl_pa.DataFrameSchema(
            {
                "a": pl_pa.Column(int),
                "b": pl_pa.Column(str, nullable=True),
            },
            strict=True,
        )
        y = polars_io.to_yaml(schema, minimal=True)
        assert polars_io.from_yaml(y) == schema
        j = polars_io.to_json(schema, minimal=True)
        assert polars_io.from_json(j) == schema

    def test_minimal_omits_version(self) -> None:
        import pandera.polars as pl_pa
        from pandera.io import polars_io

        schema = pl_pa.DataFrameSchema({"a": pl_pa.Column(int)})
        mini = polars_io.serialize_schema(schema, minimal=True)
        assert "version" not in mini
        full = polars_io.serialize_schema(schema, minimal=False)
        assert "version" in full

    def test_minimal_strips_default_check_options(self) -> None:
        import pandera.polars as pl_pa
        from pandera.io import polars_io

        schema = pl_pa.DataFrameSchema(
            {"a": pl_pa.Column(int, checks=pl_pa.Check.ge(0))},
        )
        y = polars_io.to_yaml(schema, minimal=True)
        assert "ignore_na" not in y
        assert polars_io.from_yaml(y) == schema


@pytest.mark.parametrize("backend", ["ibis", "pyspark"])
class TestIbisPysparkSerdesMinimal:
    """Ibis and PySpark SQL IO share the same serdes shape."""

    def test_roundtrip_and_minimal_version(self, backend: str) -> None:
        io_mod: ModuleType
        if backend == "ibis":
            pytest.importorskip("ibis")
            import pandera.ibis as ibis_pa
            from pandera.io import ibis_io

            schema = ibis_pa.DataFrameSchema(
                {
                    "a": ibis_pa.Column(int),
                    "b": ibis_pa.Column(str, nullable=True),
                },
                strict=True,
            )
            io_mod = ibis_io
        else:
            pytest.importorskip("pyspark")
            import pandera.pyspark as pyspark_pa
            from pandera.io import pyspark_sql_io

            schema = pyspark_pa.DataFrameSchema(
                {
                    "a": pyspark_pa.Column("long"),
                    "b": pyspark_pa.Column("string", nullable=True),
                },
                strict=True,
            )
            io_mod = pyspark_sql_io

        assert io_mod.from_yaml(io_mod.to_yaml(schema, minimal=True)) == schema
        assert io_mod.from_json(io_mod.to_json(schema, minimal=True)) == schema

        mini = io_mod.serialize_schema(schema, minimal=True)
        assert "version" not in mini
        full = io_mod.serialize_schema(schema, minimal=False)
        assert "version" in full


class TestXarraySerdesMinimal:
    """Xarray :mod:`pandera.io.xarray_io` minimal serialization."""

    def test_data_array_minimal_omits_version_roundtrips(self) -> None:
        pytest.importorskip("xarray")
        import numpy as np

        from pandera.api.xarray.container import DataArraySchema
        from pandera.io import xarray_io

        schema = DataArraySchema(
            dtype=np.float64,
            dims=("x",),
            name="v",
            nullable=True,
        )
        mini = xarray_io.serialize_data_array_schema(schema, minimal=True)
        assert "version" not in mini
        assert mini["nullable"] is True
        restored = xarray_io.deserialize_data_array_schema(mini)
        assert restored.dims == schema.dims
        assert restored.name == schema.name
        assert restored.nullable is True
        assert str(restored.dtype) == str(schema.dtype)

        y = xarray_io.to_yaml(schema, minimal=True)
        ry = xarray_io.from_yaml(y)
        assert ry.dims == schema.dims and ry.name == schema.name
        j = xarray_io.to_json(schema, minimal=True)
        rj = xarray_io.from_json(j)
        assert rj.dims == schema.dims and rj.name == schema.name

    def test_dataset_minimal_roundtrip(self) -> None:
        pytest.importorskip("xarray")

        from pandera.api.xarray.components import DataVar
        from pandera.api.xarray.container import DatasetSchema
        from pandera.io import xarray_io

        schema = DatasetSchema(
            data_vars={"t": DataVar(dtype="float64", dims=("x",))},
            strict=True,
        )
        mini = xarray_io.serialize_dataset_schema(schema, minimal=True)
        assert "version" not in mini
        assert mini["strict"] is True
        restored = xarray_io.deserialize_dataset_schema(mini)
        assert restored.strict is True
        assert "t" in restored.data_vars
        ry = xarray_io.from_yaml(xarray_io.to_yaml(schema, minimal=True))
        assert ry.strict is True
        assert "t" in ry.data_vars
