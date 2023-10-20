"""Unit tests for the geopandas integration."""

import shapely
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point

import pandera as pa
from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


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
    # pylint: disable=missing-class-docstring
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
    "data,invalid",
    [
        [
            gpd.GeoDataFrame(
                {"geometry": [Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))] * 2},
                crs="EPSG:25832",
            ),
            True,
        ],
    ],
)
def test_schema_from_dataframe(data, invalid: bool):
    # pylint: disable=missing-class-docstring
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
            GeoDataFrame[Schema](data)
        return

    assert isinstance(GeoDataFrame[Schema](data), gpd.GeoDataFrame)


@pytest.mark.parametrize(
    "data,invalid",
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
            False,
        ],
        [["POINT (0 0)", "POINT (1 1)"], False],
        [[Point(0, 0), Point(1, 1)], False],
        [shapely.to_wkb(shapely.points([[0, 0], [1, 1]])), False],
        [shapely.points([[0, 0], [1, 1]]), False],
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
            True,
        ],
    ],
)
def test_schema_coerce_2d(data, invalid: bool):
    # pylint: disable=missing-class-docstring
    """Test that DataFrameModel works on GeoDataFrames."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            coerce = True

        @pa.check("geometry")
        @classmethod
        def geo_check(cls, geo_series: GeoSeries) -> Series[bool]:
            # pylint: disable=missing-function-docstring
            return ~geo_series.has_z

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(pa.errors.SchemaError):
            GeoDataFrame[Schema]({"geometry": data})
        return

    assert isinstance(
        GeoDataFrame[Schema]({"geometry": data}), gpd.GeoDataFrame
    )


@pytest.mark.parametrize(
    "data,invalid",
    [
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
            False,
        ],
        [["POINT (0 0 0)", "POINT (1 1 1)"], False],
        [[Point(0, 0, 0), Point(1, 1, 1)], False],
        [shapely.to_wkb(shapely.points([[0, 0, 0], [1, 1, 1]])), False],
        [shapely.points([[0, 0, 0], [1, 1, 1]]), False],
    ],
)
def test_schema_coerce_3d(data, invalid: bool):
    # pylint: disable=missing-class-docstring
    """Test that DataFrameModel works on GeoDataFrames."""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            coerce = True

        @pa.check("geometry")
        @classmethod
        def geo_check(cls, geo_series: GeoSeries) -> Series[bool]:
            # pylint: disable=missing-function-docstring
            return geo_series.has_z

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(pa.errors.SchemaError):
            GeoDataFrame[Schema]({"geometry": data})
        return

    assert isinstance(
        GeoDataFrame[Schema]({"geometry": data}), gpd.GeoDataFrame
    )
