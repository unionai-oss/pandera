"""Tests for GeoPandas IO (to/from YAML/JSON) and schema inference."""

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import pandera
import pandera.geopandas as pg
from pandera.api.geopandas.container import GeoDataFrameSchema
from pandera.typing.geopandas import GeoDataFrame, GeoSeries

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _make_geo_schema():
    """Create a simple GeoDataFrameSchema for testing IO round-trips."""
    return GeoDataFrameSchema(
        columns={
            "geometry": pg.Column("geometry"),
            "name": pg.Column(str),
            "value": pg.Column(
                int,
                checks=[pg.Check.greater_than(0)],
            ),
        },
        coerce=True,
    )


def _make_geodataframe():
    """Create a sample GeoDataFrame for inference tests."""
    return gpd.GeoDataFrame(
        {
            "geometry": gpd.GeoSeries(
                [Point(0, 0), Point(1, 1)],
            ),
            "name": ["a", "b"],
            "value": [1, 2],
        }
    )


# --------------------------------------------------------------------------- #
# GeoDataFrameSchema.to_yaml / from_yaml
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
class TestGeoSchemaYaml:
    def test_to_yaml_returns_string(self):
        schema = _make_geo_schema()
        yaml_str = schema.to_yaml()
        assert isinstance(yaml_str, str)
        assert "geometry" in yaml_str

    def test_round_trip_string(self):
        schema = _make_geo_schema()
        yaml_str = schema.to_yaml()
        loaded = GeoDataFrameSchema.from_yaml(yaml_str)
        assert isinstance(loaded, GeoDataFrameSchema)
        assert set(loaded.columns) == set(schema.columns)

    def test_round_trip_file(self, tmp_path):
        schema = _make_geo_schema()
        fp = tmp_path / "geo_schema.yaml"
        schema.to_yaml(fp)
        loaded = GeoDataFrameSchema.from_yaml(fp)
        assert isinstance(loaded, GeoDataFrameSchema)
        assert set(loaded.columns) == set(schema.columns)

    def test_loaded_schema_validates_geodataframe(self):
        schema = _make_geo_schema()
        yaml_str = schema.to_yaml()
        loaded = GeoDataFrameSchema.from_yaml(yaml_str)
        gdf = _make_geodataframe()
        result = loaded.validate(gdf)
        assert isinstance(result, gpd.GeoDataFrame)


# --------------------------------------------------------------------------- #
# GeoDataFrameSchema.to_json / from_json
# --------------------------------------------------------------------------- #


class TestGeoSchemaJson:
    def test_to_json_returns_string(self):
        schema = _make_geo_schema()
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "columns" in parsed
        assert "geometry" in parsed["columns"]

    def test_round_trip_string(self):
        schema = _make_geo_schema()
        json_str = schema.to_json()
        loaded = GeoDataFrameSchema.from_json(json_str)
        assert isinstance(loaded, GeoDataFrameSchema)
        assert set(loaded.columns) == set(schema.columns)

    def test_round_trip_file(self, tmp_path):
        schema = _make_geo_schema()
        fp = tmp_path / "geo_schema.json"
        schema.to_json(fp)
        loaded = GeoDataFrameSchema.from_json(fp)
        assert isinstance(loaded, GeoDataFrameSchema)
        assert set(loaded.columns) == set(schema.columns)

    def test_loaded_schema_validates_geodataframe(self):
        schema = _make_geo_schema()
        json_str = schema.to_json()
        loaded = GeoDataFrameSchema.from_json(json_str)
        gdf = _make_geodataframe()
        result = loaded.validate(gdf)
        assert isinstance(result, gpd.GeoDataFrame)


# --------------------------------------------------------------------------- #
# GeoDataFrameModel IO
# --------------------------------------------------------------------------- #


class _SampleGeoModel(pg.GeoDataFrameModel):
    geometry: GeoSeries
    name: str
    value: int

    class Config:
        coerce = True


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
class TestGeoModelYaml:
    def test_to_yaml_returns_string(self):
        yaml_str = _SampleGeoModel.to_yaml()
        assert isinstance(yaml_str, str)
        assert "geometry" in yaml_str

    def test_round_trip_returns_geo_schema(self):
        yaml_str = _SampleGeoModel.to_yaml()
        loaded = _SampleGeoModel.from_yaml(yaml_str)
        assert isinstance(loaded, GeoDataFrameSchema)
        assert set(loaded.columns) == {"geometry", "name", "value"}

    def test_round_trip_file(self, tmp_path):
        fp = tmp_path / "model.yaml"
        _SampleGeoModel.to_yaml(fp)
        loaded = _SampleGeoModel.from_yaml(fp)
        assert isinstance(loaded, GeoDataFrameSchema)

    def test_loaded_schema_validates(self):
        yaml_str = _SampleGeoModel.to_yaml()
        loaded = _SampleGeoModel.from_yaml(yaml_str)
        gdf = _make_geodataframe()
        result = loaded.validate(gdf)
        assert isinstance(result, gpd.GeoDataFrame)


class TestGeoModelJson:
    def test_to_json_returns_string(self):
        json_str = _SampleGeoModel.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "geometry" in parsed["columns"]

    def test_round_trip_returns_geo_schema(self):
        json_str = _SampleGeoModel.to_json()
        loaded = _SampleGeoModel.from_json(json_str)
        assert isinstance(loaded, GeoDataFrameSchema)
        assert set(loaded.columns) == {"geometry", "name", "value"}

    def test_round_trip_file(self, tmp_path):
        fp = tmp_path / "model.json"
        _SampleGeoModel.to_json(fp)
        loaded = _SampleGeoModel.from_json(fp)
        assert isinstance(loaded, GeoDataFrameSchema)

    def test_loaded_schema_validates(self):
        json_str = _SampleGeoModel.to_json()
        loaded = _SampleGeoModel.from_json(json_str)
        gdf = _make_geodataframe()
        result = loaded.validate(gdf)
        assert isinstance(result, gpd.GeoDataFrame)


# --------------------------------------------------------------------------- #
# Schema Inference
# --------------------------------------------------------------------------- #


class TestInferSchema:
    def test_infer_from_geodataframe(self):
        gdf = _make_geodataframe()
        schema = pg.infer_schema(gdf)
        assert isinstance(schema, GeoDataFrameSchema)
        assert "geometry" in schema.columns
        assert "name" in schema.columns
        assert "value" in schema.columns

    def test_infer_from_plain_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        schema = pg.infer_schema(df)
        assert isinstance(schema, GeoDataFrameSchema)
        assert "a" in schema.columns
        assert "b" in schema.columns

    def test_infer_from_series(self):
        s = pd.Series([1, 2, 3], name="vals")
        schema = pg.infer_schema(s)
        from pandera.api.pandas.array import SeriesSchema

        assert isinstance(schema, SeriesSchema)

    def test_infer_validates_geodataframe(self):
        gdf = _make_geodataframe()
        schema = pg.infer_schema(gdf)
        result = schema.validate(gdf)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_infer_invalid_type(self):
        with pytest.raises(TypeError):
            pg.infer_schema("not a dataframe")

    def test_inferred_schema_to_yaml(self):
        if not HAS_YAML:
            pytest.skip("pyyaml not installed")
        gdf = _make_geodataframe()
        schema = pg.infer_schema(gdf)
        yaml_str = schema.to_yaml()
        assert isinstance(yaml_str, str)
        loaded = GeoDataFrameSchema.from_yaml(yaml_str)
        assert isinstance(loaded, GeoDataFrameSchema)

    def test_inferred_schema_to_json(self):
        gdf = _make_geodataframe()
        schema = pg.infer_schema(gdf)
        json_str = schema.to_json()
        loaded = GeoDataFrameSchema.from_json(json_str)
        assert isinstance(loaded, GeoDataFrameSchema)
