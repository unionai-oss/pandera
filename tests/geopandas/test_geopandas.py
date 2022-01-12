"""Unit tests for the geopandas integration."""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

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
        [[Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))] * 2, True],
        [[Polygon(((0, 0), (0, -2), (-2, -2), (-2, 0)))] * 2, False],
    ],
)
def test_schema_model(data, invalid: bool):
    # pylint: disable=missing-class-docstring
    """Test that SchemaModel works on GeoDataFrames."""

    class Schema(pa.SchemaModel):
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
