"""Unit tests for the geopandas engine dtype Geometry."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point

import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime
from pandera.engines.geopandas_engine import Geometry


def test_engine_geometry_simple():
    """Test Geometry for basic attributes."""
    dtype = Geometry(crs=None)
    assert dtype.crs is None
    assert str(dtype) == "geometry"

    dtype = Geometry(crs="EPSG:4326")
    assert dtype.crs == "EPSG:4326"


def test_engine_geometry_equality():
    """Test Geometry for equivalency to other Geometry."""
    dtype = Geometry(crs="EPSG:4326")
    assert dtype == Geometry(crs="EPSG:4326")
    assert dtype != Geometry(crs="EPSG:3857")

    with pytest.raises(TypeError):
        Geometry(crs="this is definitely not a valid crs")


@pytest.mark.parametrize(
    "pandera_dtype,data_container,invalid",
    [
        [Geometry(crs="EPSG:4326"), None, False],
        [Geometry(crs="EPSG:25832"), None, "fail"],
        [DateTime, None, "fail"],
        [
            Geometry(crs="EPSG:4326"),
            gpd.GeoSeries([Point([0, 1e6])], crs="EPSG:4326"),
            False,
        ],
        [
            Geometry(crs="EPSG:4326"),
            gpd.GeoSeries([Point([0, 1e6])], crs="EPSG:25832"),
            "exception",
        ],
        [
            Geometry(crs="EPSG:4326"),
            gpd.GeoDataFrame(
                {
                    "geometry": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:4326"
                    ),
                    "random": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:4326"
                    ),
                }
            ),
            False,
        ],
        [
            Geometry(crs="EPSG:25832"),
            gpd.GeoDataFrame(
                {
                    "geometry": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:25832"
                    ),
                    "random": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:25832"
                    ),
                }
            ),
            "exception",
        ],
        [
            Geometry(crs="EPSG:4326"),
            gpd.GeoDataFrame(
                {
                    "geometry": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:4326"
                    ),
                    "random": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:25832"
                    ),
                }
            ),
            "exception",
        ],
        [
            Geometry(crs="EPSG:25832"),
            gpd.GeoDataFrame(
                {
                    "geometry": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:25832"
                    ),
                    "random": gpd.GeoSeries(
                        [Point([5e5, 1e6])], crs="EPSG:4326"
                    ),
                }
            ),
            "exception",
        ],
    ],
)
def test_engine_geometry_check(pandera_dtype, data_container, invalid):
    """Test Geometry for dtype match on data container."""

    dtype = Geometry(crs="EPSG:4326")

    if invalid == "exception":
        with pytest.raises(TypeError):
            dtype.check(pandera_dtype, data_container)
        return
    if invalid == "fail":
        assert not np.any(dtype.check(pandera_dtype, data_container))


@pytest.mark.parametrize(
    "data_container",
    [
        gpd.GeoSeries([Point([0, 1e6])], crs="EPSG:25832"),
        gpd.GeoDataFrame(
            {
                "geometry": gpd.GeoSeries(
                    [Point([5e5, 1e6])], crs="EPSG:25832"
                ),
                "random": gpd.GeoSeries([Point([5e5, 1e6])], crs="EPSG:25832"),
            }
        ),
        gpd.GeoDataFrame(
            {
                "geometry": gpd.GeoSeries(
                    [Point([5e5, 1e6])], crs="EPSG:25832"
                ),
                "random": gpd.GeoSeries([Point([5e5, 1e6])], crs="EPSG:3857"),
            }
        ),
        pd.DataFrame(
            {
                "geometry": gpd.GeoSeries(
                    [Point([5e5, 1e6])], crs="EPSG:25832"
                ),
                "random": gpd.GeoSeries([Point([5e5, 1e6])], crs="EPSG:25832"),
            }
        ),
    ],
)
def test_engine_geometry_coerce_crs(data_container):
    """Test Geometry coerce for GeoSeries CRS reprojection transform."""

    dtype = Geometry(crs="EPSG:4326")
    coerced = dtype.coerce(data_container)
    assert not np.any(
        shapely.equals_exact(
            data_container.to_numpy(),
            coerced.to_numpy(),
            tolerance=1e-3,
        )
    )
    assert np.all(coerced.crs == dtype.crs)


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
def test_engine_geometry_coerce_data(data, dims: int, invalid: bool):
    """Test Geometry input parsing."""
    series = pd.Series(data)
    dtype = Geometry()

    if invalid:
        with pytest.raises((pa.errors.SchemaError, pa.errors.ParserError)):
            dtype.coerce(series)
        return

    coerced = dtype.coerce(series)
    assert isinstance(coerced, gpd.GeoSeries)

    check_2d = np.all(shapely.has_z(coerced)) and dims == 3
    check_3d = not np.any(shapely.has_z(coerced)) and dims == 2
    assert check_2d or check_3d
