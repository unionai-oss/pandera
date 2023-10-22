"""Unit tests for the geopandas integration."""

from typing import Annotated
import shapely
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point
from pydantic import BaseModel

import pandera as pa
from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries
from pandera.engines.pandas_engine import Geometry


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


def test_pydantic_model():
    """Test that GeoDataFrame type can be used in a Pydantic model"""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

    class MyModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[Schema]

    obj = MyModel(data=gpd.GeoDataFrame({"geometry": [Point(0, 0)]}))

    assert isinstance(obj.data, gpd.GeoDataFrame)


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
    ],
)
def test_schema_dtype_without_crs(gdf_args, invalid: bool):
    """Test Geometry without CRS."""
    # No CRS to validate
    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry()  # type: ignore

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(TypeError):
            GeoDataFrame[Schema](**gdf_args)
        return

    assert isinstance(GeoDataFrame[Schema](**gdf_args), gpd.GeoDataFrame)


def test_schema_dtype_crs_transform():
    """Test Geometry CRS coerce for coordinate transform."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry(crs="EPSG:4326")

        class Config:
            coerce = True

    gdf = GeoDataFrame[Schema](
        data={"geometry": [Point([0, 1e6])]}, crs="EPSG:25832"
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == "EPSG:4326"
    golden_gs = gpd.GeoSeries(data=[Point([4.4553, 9.0184])], crs="EPSG:4326")
    golden_compare = gdf.geometry.geom_equals_exact(golden_gs, tolerance=1e-3)
    assert golden_compare.all()


def test_schema_dtype_parametrized_crs():
    """Test Geometry declaration using dtype_kwargs and Annotated."""

    gdf = gpd.GeoDataFrame({"geometry": [Point([1, 1])]}, crs="EPSG:4326")

    class Schema1(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pa.Field(dtype_kwargs={"crs": "EPSG:4326"})

    assert isinstance(GeoDataFrame[Schema1](gdf), gpd.GeoDataFrame)

    class Schema2(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Annotated[GeoSeries, "EPSG:4326"]

    assert isinstance(GeoDataFrame[Schema2](gdf), gpd.GeoDataFrame)


def test_schema_dtype_invalid_crs():
    """Test Geometry for invalid CRS."""
    with pytest.raises(TypeError):
        Geometry(crs="this is definitely not a valid crs")


def test_schema_dtype_multiple_crs():
    """Test GeoDataFrame with multiple GeoSeries columns on different CRS"""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry(crs="EPSG:4326")
        random: Geometry(crs="EPSG:3857")

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
    """Test that DataFrameModel works on gpd.GeoDataFrame input."""

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


@pytest.mark.parametrize(
    "data,dims,invalid",
    [
        [
            [
                {"type": "Point", "coordinates": [139.86681009, 35.77565643]},
                {
                    "type": "LineString",
                    "coordinates": [
                        [139.86681009, 35.77565643],
                        [139.86677824, 35.7756761],
                    ],
                },
                {
                    "type": "LineString",
                    "coordinates": [
                        [139.86677824, 35.7756761],
                        [139.86676329, 35.77568168],
                    ],
                },
            ],
            2,
            False,
        ],
        [["POINT (0 0)", "POINT (1 1)"], 2, False],
        [[Point(0, 0), Point(1, 1)], 2, False],
        [shapely.to_wkb(shapely.points([[0, 0], [1, 1]])), 2, False],
        [shapely.points([[0, 0], [1, 1]]), 2, False],
        [[1, 2], 2, True],
        [
            [
                {
                    "type": "InvalidPoint!",
                    "coordinates": [139.86681009, 35.77565643],
                },
                {
                    "type": "LineString",
                    "coordinates": [
                        [139.86681009, 35.77565643],
                        [139.86677824, 35.7756761],
                    ],
                },
            ],
            2,
            True,
        ],
        [
            [
                {
                    "type": "Point",
                    "coordinates": [139.86681009, 35.77565643, 9.031],
                },
                {
                    "type": "LineString",
                    "coordinates": [
                        [139.86681009, 35.77565643, 9.031],
                        [139.86677824, 35.7756761, 9.037],
                    ],
                },
                {
                    "type": "LineString",
                    "coordinates": [
                        [139.86677824, 35.7756761, 9.037],
                        [139.86676329, 35.77568168, 9.041],
                    ],
                },
            ],
            3,
            False,
        ],
        [["POINT (0 0 0)", "POINT (1 1 1)"], 3, False],
        [[Point(0, 0, 0), Point(1, 1, 1)], 3, False],
        [shapely.to_wkb(shapely.points([[0, 0, 0], [1, 1, 1]])), 3, False],
        [shapely.points([[0, 0, 0], [1, 1, 1]]), 3, False],
    ],
)
def test_schema_coerce_input(data, dims: int, invalid: bool):
    """Test 3D Geometry input parsing."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            coerce = True

        @pa.check("geometry")
        @classmethod
        def geo_check(cls, geo_series: GeoSeries) -> Series[bool]:
            # pylint: disable=missing-function-docstring
            return ((dims == 3) & geo_series.has_z) | (
                (dims == 2) & ~geo_series.has_z
            )

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(pa.errors.SchemaError):
            GeoDataFrame[Schema]({"geometry": data})
        return

    assert isinstance(
        GeoDataFrame[Schema]({"geometry": data}), gpd.GeoDataFrame
    )
