"""Pandera type annotations for GeoPandas."""
from pandera.engines.pandas_engine import Geometry
from pandera.typing.pandas import DataFrame, Series, T

try:
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

if GEOPANDAS_INSTALLED:
    # pylint:disable=too-few-public-methods
    class GeoSeries(Series[Geometry], gpd.GeoSeries):
        """Representation of geopandas.GeoSeries, only used for type annotation."""

    class GeoDataFrame(DataFrame[T], gpd.GeoDataFrame):
        """Representation of geopandas.GeoDataFrame, only used for type annotation."""
