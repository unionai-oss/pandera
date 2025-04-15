"""Tests GeoPandas schema creation and validation from type annotations."""

# pylint:disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import geopandas as gpd
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError
from shapely.geometry import Point

import pandera.pandas as pa
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


def test_pydantic_active_geometry():
    """Test that GeoDataFrame type can be used in a Pydantic model with geometry activated"""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

    class MyModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[Schema]

    # gpd.GeoDataFrame input
    obj = MyModel(
        data=gpd.GeoDataFrame(
            {
                "geometry": gpd.GeoSeries([Point(0, 0)]),
            }
        )
    )

    assert isinstance(obj.data, gpd.GeoDataFrame)
    assert obj.data.geometry.name == "geometry"

    # pd.DataFrame input (coerce to gpd.GeoDataFrame)
    obj = MyModel(
        data=pd.DataFrame(
            {
                "geometry": gpd.GeoSeries([Point(0, 0)]),
            }
        )
    )

    assert isinstance(obj.data, gpd.GeoDataFrame)


def test_pydantic_inactive_geometry():
    """Test that GeoDataFrame type can be used in a Pydantic model with geometry not activated"""

    # Geometry column exists but non-standard name
    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        random: GeoSeries

    class MyModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[Schema]

    obj = MyModel(
        data=pd.DataFrame(
            {
                "random": gpd.GeoSeries([Point(0, 0)]),
            }
        )
    )

    assert isinstance(obj.data, gpd.GeoDataFrame)

    with pytest.raises(
        AttributeError,
        match="the active geometry column to use has not been set",
    ):
        _ = obj.data.geometry

    # Geometry column doesn't exist
    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        random: str

    class MyModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[Schema]

    obj = MyModel(
        data=pd.DataFrame(
            {
                "random": ["a", "b"],
            }
        )
    )

    assert isinstance(obj.data, gpd.GeoDataFrame)

    with pytest.raises(
        AttributeError,
        match="the active geometry column to use has not been set",
    ):
        _ = obj.data.geometry


def test_pydantic_garbage_input():
    """Test that GeoDataFrame type in a Pydantic model will throw an exception with garbage input"""

    class Schema(pa.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

    class MyModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[Schema]

    with pytest.raises(
        ValidationError,
        match="Expected gpd.GeoDataFrame",
    ):
        MyModel(data="invalid")
