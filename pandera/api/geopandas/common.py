"""Shared helpers for :mod:`pandera.geopandas` (schema and model APIs)."""

from __future__ import annotations

import pandas as pd


def require_geopandas() -> None:
    from pandera.typing import geopandas as gp_typing

    if not gp_typing.GEOPANDAS_INSTALLED:
        raise ImportError(
            "This API requires geopandas. Install pandera with the `geopandas` "
            "extra, e.g. `pip install pandera[geopandas]`."
        )


def to_geodataframe(obj: pd.DataFrame) -> pd.DataFrame:
    """Return ``obj`` as a :class:`geopandas.GeoDataFrame` when needed."""
    require_geopandas()
    import geopandas as gpd

    from pandera.typing.geopandas import GeoDataFrame

    if isinstance(obj, gpd.GeoDataFrame):
        return obj
    return GeoDataFrame._coerce_geometry(obj)
