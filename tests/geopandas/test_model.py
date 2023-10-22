"""Tests GeoPandas schema creation and validation from type annotations."""
# pylint:disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
from typing import Optional

import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

import pandera as pa
from pandera.typing import Index, Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


def test_from_records_validates_the_schema():
    """Test that GeoDataFrame[Schema] validates the schema"""

    class Schema(pa.DataFrameModel):
        geometry: GeoSeries
        state: Series[str]
        city: Series[str]
        price: Series[float]
        postal_code: Optional[Series[int]] = pa.Field(nullable=True)

    raw_data = [
        {
            "geometry": Point(0, 0),
            "state": "NY",
            "city": "New York",
            "price": 8.0,
        },
        {
            "geometry": Point(1, 1),
            "state": "FL",
            "city": "Miami",
            "price": 12.0,
        },
    ]
    pandera_validated_df = GeoDataFrame.from_records(Schema, raw_data)
    pandas_df = gpd.GeoDataFrame(pd.DataFrame.from_records(raw_data))
    assert pandera_validated_df.equals(Schema.validate(pandas_df))
    assert isinstance(pandera_validated_df, GeoDataFrame)
    assert isinstance(pandas_df, gpd.GeoDataFrame)

    raw_data = [
        {
            "geometry": Point(0, 0),
            "state": "NY",
            "city": "New York",
        },
        {
            "geometry": Point(1, 1),
            "state": "FL",
            "city": "Miami",
        },
    ]

    with pytest.raises(
        pa.errors.SchemaError,
        match="^column 'price' not in dataframe",
    ):
        GeoDataFrame[Schema](raw_data)


def test_from_records_sets_the_index_from_schema():
    """Test that GeoDataFrame[Schema] validates the schema"""

    class Schema(pa.DataFrameModel):
        geometry: GeoSeries
        state: Index[str] = pa.Field(check_name=True)
        city: Series[str]
        price: Series[float]

    raw_data = [
        {
            "geometry": Point(0, 0),
            "state": "NY",
            "city": "New York",
            "price": 8.0,
        },
        {
            "geometry": Point(1, 1),
            "state": "FL",
            "city": "Miami",
            "price": 12.0,
        },
    ]
    pandera_validated_df = GeoDataFrame.from_records(Schema, raw_data)
    pandas_df = gpd.GeoDataFrame(
        pd.DataFrame.from_records(raw_data, index=["state"])
    )
    assert pandera_validated_df.equals(Schema.validate(pandas_df))
    assert isinstance(pandera_validated_df, GeoDataFrame)
    assert isinstance(pandas_df, gpd.GeoDataFrame)


def test_from_records_sorts_the_columns():
    """Test that GeoDataFrame[Schema] validates the schema"""

    class Schema(pa.DataFrameModel):
        geometry: GeoSeries
        state: Series[str]
        city: Series[str]
        price: Series[float]

    raw_data = [
        {
            "geometry": Point(0, 0),
            "city": "New York",
            "price": 8.0,
            "state": "NY",
        },
        {
            "geometry": Point(1, 1),
            "price": 12.0,
            "state": "FL",
            "city": "Miami",
        },
    ]
    pandera_validated_df = GeoDataFrame.from_records(Schema, raw_data)
    pandas_df = gpd.GeoDataFrame(
        pd.DataFrame.from_records(raw_data)[
            ["geometry", "state", "city", "price"]
        ]
    )
    assert pandera_validated_df.equals(Schema.validate(pandas_df))
    assert isinstance(pandera_validated_df, GeoDataFrame)
    assert isinstance(pandas_df, gpd.GeoDataFrame)
