"""GeoPandas validation API (schemas and models)."""

from pandera.api.geopandas.container import GeoDataFrameSchema
from pandera.api.geopandas.model import GeoDataFrameModel

__all__ = ["GeoDataFrameModel", "GeoDataFrameSchema"]
