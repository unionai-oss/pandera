"""Tests GeoPandas schema creation and validation from type annotations."""

import geopandas as gpd
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError
from shapely.geometry import Point

import pandera.geopandas as pg
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


def test_pydantic_active_geometry():
    """Test that GeoDataFrame type can be used in a Pydantic model with geometry activated"""

    class Schema(pg.DataFrameModel):
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
    class Schema(pg.DataFrameModel):
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
    class Schema(pg.DataFrameModel):
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


def test_pydantic_geodataframe_model_schema():
    """GeoDataFrame[Schema] works when Schema subclasses GeoDataFrameModel."""

    class GeoSchema(pg.GeoDataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            coerce = True

    class MyModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[GeoSchema]

    obj = MyModel(
        data=pd.DataFrame({"geometry": [Point(0, 0)]}),
    )
    assert isinstance(obj.data, gpd.GeoDataFrame)


def test_pydantic_garbage_input():
    """Test that GeoDataFrame type in a Pydantic model will throw an exception with garbage input"""

    class Schema(pg.DataFrameModel):
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


def test_pydantic_collected_schema_errors():
    """Collected schema errors must surface as a ``ValidationError``.

    ``SchemaErrors`` is a sibling of ``SchemaError``, not a subclass, and the
    pydantic hook converted only the latter, so the first two models below
    raised the pandera error straight out of the model constructor. The last
    case pins the single-error path from the same file.
    """

    class StrictSchema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            strict = True

    class CoerceSchema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries
        a: GeoSeries

        class Config:
            coerce = True

    class PlainSchema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries
        a: GeoSeries

    class StrictModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[StrictSchema]

    class CoerceModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[CoerceSchema]

    class PlainModel(BaseModel):
        # pylint: disable=missing-class-docstring
        data: GeoDataFrame[PlainSchema]

    geometry = gpd.GeoSeries([Point(0, 0)])

    with pytest.raises(ValidationError, match="COLUMN_NOT_IN_SCHEMA"):
        StrictModel(
            data=gpd.GeoDataFrame({"geometry": geometry, "extra": [1]})
        )

    with pytest.raises(ValidationError, match="DATATYPE_COERCION"):
        CoerceModel(
            data=gpd.GeoDataFrame({"geometry": geometry, "a": ["nope"]})
        )

    with pytest.raises(ValidationError, match="expected series"):
        PlainModel(
            data=gpd.GeoDataFrame({"geometry": geometry, "a": ["nope"]})
        )
