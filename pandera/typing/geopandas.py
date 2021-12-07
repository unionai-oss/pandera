import geopandas as gpd
from pandera.typing.pandas import T, Series, DataFrame

try:
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

if GEOPANDAS_INSTALLED:
    GeoSeries = Series[gpd.array.GeometryDtype]

    class GeoDataFrame(DataFrame[T], gpd.GeoDataFrame):
        """Representation of geopandas.GeoDataFrame, only used for type annotation."""
        pass
