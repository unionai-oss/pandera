"""Unit tests for the geopandas integration."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import pandera.geopandas as pg
from pandera.engines.geopandas_engine import Geometry
from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries

try:  # python 3.9+
    from typing import Annotated  # type: ignore
except ImportError:
    from typing import Annotated  # type: ignore


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
        pg.DataFrameSchema({"geometry": pg.Column("geometry")}),
        pg.DataFrameSchema({"geometry": pg.Column(gpd.array.GeometryDtype)}),
        pg.DataFrameSchema({"geometry": pg.Column(gpd.array.GeometryDtype())}),
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

    class Schema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        @pg.check("geometry")
        @classmethod
        def geo_check(cls, geo_series: GeoSeries) -> Series[bool]:
            # pylint: disable=missing-function-docstring
            return geo_series.area >= 2

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises(pg.errors.SchemaError):
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
    class Schema(pg.DataFrameModel):
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
    class Schema(pg.DataFrameModel):
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

    class Schema1(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pg.Field(dtype_kwargs={"crs": "EPSG:4326"})

    assert isinstance(GeoDataFrame[Schema1](gdf), gpd.GeoDataFrame)

    class Schema2(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Annotated[Geometry, "EPSG:4326"]

    assert isinstance(GeoDataFrame[Schema2](gdf), gpd.GeoDataFrame)


def test_schema_multiple_geometry_same_crs():
    """Test GeoDataFrame with multiple GeoSeries columns on same CRS"""

    class Schema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pg.Field(dtype_kwargs={"crs": "EPSG:4326"})
        random: Geometry = pg.Field(dtype_kwargs={"crs": "EPSG:4326"})

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

    class Schema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: Geometry = pg.Field(dtype_kwargs={"crs": "EPSG:4326"})
        random: Geometry = pg.Field(dtype_kwargs={"crs": "EPSG:3857"})

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

    class Schema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

        class Config:
            coerce = True

    # create a geodataframe that's validated on object initialization
    if invalid:
        with pytest.raises((pg.errors.SchemaError, pg.errors.SchemaErrors)):
            GeoDataFrame[Schema](data)
        return

    assert isinstance(GeoDataFrame[Schema](data), gpd.GeoDataFrame)


def test_schema_no_geometry():
    """Test that GeoDataFrame can be constructed from data without a Geometry column."""

    class Schema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        name: str

    # create a geodataframe that's validated on object initialization
    assert isinstance(
        GeoDataFrame[Schema]({"name": ["a", "b"]}), gpd.GeoDataFrame
    )


def test_geodataframe_model_validate_returns_geodataframe():
    """GeoDataFrameModel.validate always yields a geopandas.GeoDataFrame."""

    class GSchema(pg.GeoDataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries
        region: Series[str]

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
            ],
            "region": ["NA"],
        },
        crs="EPSG:4326",
    )
    out = GSchema.validate(gdf)
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == gdf.crs

    pdf = pd.DataFrame(
        {
            "geometry": gdf["geometry"],
            "region": gdf["region"],
        }
    )
    out_pdf = GSchema.validate(pdf)
    assert isinstance(out_pdf, gpd.GeoDataFrame)


def test_geodataframe_model_vs_dataframe_model_return_type():
    """DataFrameModel may return a plain DataFrame; GeoDataFrameModel coerces."""

    data = {
        "geometry": [
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        ],
        "name": ["a"],
    }

    class PSchema(pg.DataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries
        name: str

        class Config:
            coerce = True

    class GSchema(pg.GeoDataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries
        name: str

        class Config:
            coerce = True

    pdf = pd.DataFrame(data)
    p_out = PSchema.validate(pdf)
    g_out = GSchema.validate(pdf)
    assert type(p_out) is pd.DataFrame
    assert isinstance(g_out, gpd.GeoDataFrame)


def test_geodataframe_model_init_alias():
    """Calling GeoDataFrameModel(...) delegates to validate like DataFrameModel."""

    class GSchema(pg.GeoDataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries

    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))]},
        crs="EPSG:4326",
    )
    assert isinstance(GSchema(gdf), gpd.GeoDataFrame)


def test_geodataframe_model_with_typing_generic():
    """GeoDataFrame[GSchema] works when GSchema subclasses GeoDataFrameModel."""

    class GSchema(pg.GeoDataFrameModel):
        # pylint: disable=missing-class-docstring
        geometry: GeoSeries
        x: Series[int]

    gdf = GeoDataFrame[GSchema](
        {
            "geometry": [Point(0, 0)],
            "x": [1],
        }
    )
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_geodataframe_schema_validate_returns_geodataframe():
    """GeoDataFrameSchema.validate coerces plain DataFrames to GeoDataFrame."""
    schema = pg.GeoDataFrameSchema(
        {
            "geometry": pg.Column("geometry", coerce=True),
            "region": pg.Column(str),
        }
    )
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
            ],
            "region": ["NA"],
        },
        crs="EPSG:4326",
    )
    assert isinstance(schema.validate(gdf), gpd.GeoDataFrame)

    pdf = pd.DataFrame(
        {
            "geometry": gdf["geometry"],
            "region": gdf["region"],
        }
    )
    assert isinstance(schema.validate(pdf), gpd.GeoDataFrame)


def test_geodataframe_schema_vs_dataframe_schema_return_type():
    """DataFrameSchema may return a plain DataFrame; GeoDataFrameSchema coerces."""
    cols = {
        "geometry": pg.Column("geometry", coerce=True),
        "name": pg.Column(str),
    }
    pdf = pd.DataFrame(
        {
            "geometry": [Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))],
            "name": ["a"],
        }
    )
    p_out = pg.DataFrameSchema(cols).validate(pdf)
    g_out = pg.GeoDataFrameSchema(cols).validate(pdf)
    assert type(p_out) is pd.DataFrame
    assert isinstance(g_out, gpd.GeoDataFrame)


def test_geopandas_module_exports():
    """pandera.geopandas re-exports pandera.pandas plus geo schema/model."""
    import pandera.geopandas as pgeo
    import pandera.pandas as pa

    assert set(pa.__all__).issubset(set(pgeo.__all__))
    assert len(pgeo.__all__) == len(pa.__all__) + 2
    assert issubclass(pgeo.GeoDataFrameModel, pa.DataFrameModel)
    assert pgeo.Column is pa.Column
    assert pgeo.errors is pa.errors
