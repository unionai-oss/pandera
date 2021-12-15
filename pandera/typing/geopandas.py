"""Pandera type annotations for GeoPandas."""
from typing import TYPE_CHECKING, Generic, TypeVar

from pandera.typing.common import DataFrameBase, SeriesBase

from .pandas import Schema

try:
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

if GEOPANDAS_INSTALLED:
    # pylint: disable=import-outside-toplevel,ungrouped-imports
    from pandera.engines.pandas_engine import Geometry

    # pylint:disable=too-few-public-methods
    class GeoSeries(SeriesBase[Geometry], gpd.GeoSeries):
        """Representation of geopandas.GeoSeries, only used for type annotation."""

    # pylint:disable=invalid-name
    if TYPE_CHECKING:
        T = TypeVar("T")  # pragma: no cover
    else:
        T = Schema

    class GeoDataFrame(DataFrameBase, gpd.GeoDataFrame, Generic[T]):
        """Representation of geopandas.GeoDataFrame, only used for type annotation."""
