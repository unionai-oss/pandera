"""Unit tests for the geopandas integration."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import pandera.pandas as pa
from pandera.engines.geopandas_engine import Geometry
from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries

try:  # python 3.9+
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated  # type: ignore


def test_dataframe_schema():
    """Test that DataFrameSchema works on GeoDataFrames."""
    geo_df = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
            ],
        }
    )

    for geo_schema in [
        pa.DataFrameSchema({"geometry": pa.Column("geometry")}),
        pa.DataFrameSchema({"geometry": pa.Column(gpd.array.GeometryDtype)}),
        pa.DataFrameSchema({"geometry": pa.Column(gpd.array.GeometryDtype())}),
    ]:
        assert isinstance(geo_schema.validate(geo_df), gpd.GeoDataFrame)


@pytest.mark.parametrize(
    "data,invalid",
    [
        [["what the heck?!"], True],
        [[Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))] * 2, True],
        [[Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))] * 2, False],
    ],
)
def test_schema_model(data, invalid: bool):
    """Test that DataFrameModel works on GeoDataFrames."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        @pa.check("geometry")
        @classmethod
        def geo_check(cls, geo_series: GeoSeries) -> Series[bool]:
            # pylint: disable=missing-function-docstring
            return geo_series.area >= 2

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(pa.errors.SchemaError):
            GeoDataFrame[Schema]({"geometry": data})
        return

    assert isinstance(
        GeoDataFrame[Schema]({"geometry": data}), gpd.GeoDataFrame
    )


@pytest.mark.parametrize(
    "gdf_args,invalid",
    [
        [
            {
                "geometry": [Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))]
                * 2,
                "crs": None,
            },
            True,
        ],
        [
            {
                "geometry": [Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))]
                * 2,
                "crs": "EPSG:4326",
            },
            False,
        ],
        [
            {
                "geometry": [Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))]
                * 2,
                "crs": "EPSG:25832",
            },
            True,
        ],
    ],
)
def test_schema_dtype_crs_without_coerce(gdf_args, invalid: bool):
    """Test Geometry crs annotation without coerce."""

    # No CRS to validate
    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry(crs="EPSG:4326")  # type: ignore

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(TypeError):
            GeoDataFrame[Schema](**gdf_args)
        return

    assert isinstance(GeoDataFrame[Schema](**gdf_args), gpd.GeoDataFrame)


@pytest.mark.parametrize(
    "gdf_args,invalid",
    [
        [
            {
                "geometry": [Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))]
                * 2,
                "crs": None,
            },
            False,
        ],
        [
            {
                "geometry": [Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))]
                * 2,
                "crs": "EPSG:4326",
            },
            False,
        ],
        [
            {
                "geometry": [Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))]
                * 2,
                "crs": "EPSG:25832",
            },
            False,
        ],
    ],
)
def test_schema_dtype_crs_with_coerce(gdf_args, invalid: bool):
    """Test Geometry crs annotation with coerce."""

    # No CRS to validate
    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry(crs="EPSG:4326")  # type: ignore

        class Config:
            coerce = True

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(TypeError):
            GeoDataFrame[Schema](**gdf_args)
        return

    gdf = GeoDataFrame[Schema](**gdf_args)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == "EPSG:4326"


def test_schema_parametrized_crs():
    """Test Geometry declaration using dtype_kwargs and Annotated."""

    gdf = gpd.GeoDataFrame({"geometry": [Point([1, 1])]}, crs="EPSG:4326")

    class Schema1(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pa.Field(dtype_kwargs={"crs": "EPSG:4326"})

    assert isinstance(GeoDataFrame[Schema1](gdf), gpd.GeoDataFrame)

    class Schema2(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Annotated[Geometry, "EPSG:4326"]

    assert isinstance(GeoDataFrame[Schema2](gdf), gpd.GeoDataFrame)


def test_schema_multiple_geometry_same_crs():
    """Test GeoDataFrame with multiple GeoSeries columns on same CRS"""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pa.Field(dtype_kwargs={"crs": "EPSG:4326"})
        random: Geometry = pa.Field(dtype_kwargs={"crs": "EPSG:4326"})

    data = {
        "geometry": gpd.GeoSeries(
            [Point([1, 1])], name="geometry", crs="EPSG:4326"
        ),
        "random": gpd.GeoSeries(
            [Point([2, 2])], name="random", crs="EPSG:4326"
        ),
    }

    # Both columns should have same CRS
    gdf = GeoDataFrame[Schema](data)
    pd.testing.assert_series_equal(gdf["geometry"], data["geometry"])
    pd.testing.assert_series_equal(gdf["random"], data["random"])


def test_schema_multiple_geometry_different_crs():
    """Test GeoDataFrame with multiple GeoSeries columns on different CRS"""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pa.Field(dtype_kwargs={"crs": "EPSG:4326"})
        random: Geometry = pa.Field(dtype_kwargs={"crs": "EPSG:3857"})

        class Config:
            # pylint: disable=missing-class-docstring
            coerce = True

    data = {"geometry": [Point([1, 1])], "random": [Point([450000, 900000])]}

    # geometry is assigned EPSG:3857 by gpd.GeoDataFrame constructor,
    # while random isn't assigned anything. Post-coercion, both are
    # converted to their respective CRS schema.
    gdf = GeoDataFrame[Schema](data, crs="EPSG:3857")
    assert gdf.geometry.crs == "EPSG:4326"
    assert gdf["random"].crs == "EPSG:3857"

    # geometry is assigned EPSG:3395 by gpd.GeoDataFrame.to_crs, while random
    # is left unchanged
    gdf = gdf.to_crs("EPSG:3395")
    assert gdf.geometry.crs == "EPSG:3395"
    assert gdf["random"].crs == "EPSG:3857"

    # Pandera coerces geometry back to schema
    gdf = GeoDataFrame[Schema](data)
    assert gdf.geometry.crs == "EPSG:4326"
    assert gdf["random"].crs == "EPSG:3857"

    # random is assigned EPSG:4326 by gpd.GeoDataFrame.to_crs, while
    # geometry is left unchanged
    gdf["random"] = gdf["random"].to_crs("EPSG:4326")
    assert gdf.geometry.crs == "EPSG:4326"
    assert gdf["random"].crs == "EPSG:4326"

    # Pandera coerces random back to schema
    gdf = GeoDataFrame[Schema](data)
    assert gdf.geometry.crs == "EPSG:4326"
    assert gdf["random"].crs == "EPSG:3857"


@pytest.mark.parametrize(
    "data,invalid",
    [
        [
            pd.DataFrame(
                {"geometry": [Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))] * 2},
            ),
            False,
        ],
        [
            pd.DataFrame(
                {"geometry": ["a", "b"]},
            ),
            True,
        ],
        [
            gpd.GeoDataFrame(
                {"geometry": [Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))] * 2},
                crs="EPSG:25832",
            ),
            False,
        ],
    ],
)
def test_schema_from_dataframe(data, invalid: bool):
    """Test that DataFrameModel works on gpd.GeoDataFrame or pd.DataFrame input."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            coerce = True

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(pa.errors.SchemaError):
            GeoDataFrame[Schema](data)
        return

    assert isinstance(GeoDataFrame[Schema](data), gpd.GeoDataFrame)


def test_schema_no_geometry():
    """Test that GeoDataFrame can be constructed from data without a Geometry column."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        name: str

    # create a geodataframe that's validated on object initialization
    assert isinstance(
        GeoDataFrame[Schema]({"name": ["a", "b"]}), gpd.GeoDataFrame
    )
