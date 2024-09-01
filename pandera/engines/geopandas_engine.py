# pylint: disable=cyclic-import
"""Geopandas data types for the pandas type engine."""

from typing import Any, Iterable, Optional, Union

try:
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False


import dataclasses
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.geometry
from geopandas.array import GeometryArray, GeometryDtype, from_shapely

from pandera import dtypes, errors
from pandera.engines import pandas_engine

GeoPandasObject = Union[
    pd.Series, pd.DataFrame, gpd.GeoSeries, gpd.GeoDataFrame
]


@pandas_engine.Engine.register_dtype(
    equivalents=[
        "geometry",
        GeometryDtype,
        GeometryDtype(),
    ],
)
@dtypes.immutable(init=True)
class Geometry(pandas_engine.DataType):
    """Semantic representation of geopandas :class:`geopandas.array.GeometryDtype`.

    Extends the native GeometryDtype by allowing designation of a coordinate
    reference system (CRS) as found on GeometryArray, GeoSeries, and GeoDataFrame.
    When the CRS is defined, validator will check for matching CRS, and coerce
    will transform coordinate values via GeoPandas' 'to_crs' method. Otherwise, CRS
    of data is ignored.
    """

    type = GeometryDtype()

    crs: Optional[str] = dataclasses.field(default=None)
    """Coordinate Reference System of the geometry objects.
    """

    # define __init__ to please mypy
    def __init__(  # pylint:disable=super-init-not-called
        self,
        crs: Optional[Any] = None,
    ) -> None:
        if crs is not None:
            try:
                pyproj.CRS.from_user_input(crs)
            except pyproj.exceptions.CRSError as exc:
                raise TypeError(f"Invalid CRS: {str(crs)}") from exc

            object.__setattr__(self, "crs", crs)

    def _coerce_values(self, obj: GeoPandasObject) -> GeoPandasObject:
        if isinstance(obj, gpd.GeoSeries) or (
            isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame))
            and all(v == str(self) for v in obj.dtypes.to_dict().values())
        ):
            # Return as-is if we already have the proper underlying dtype
            return obj

        # Shapely objects
        try:
            return from_shapely(obj)
        except TypeError:
            ...

        # Well-known Text (WKT) strings
        try:
            return from_shapely(shapely.from_wkt(obj))
        except (TypeError, shapely.errors.GEOSException):
            ...

        # Well-known Binary (WKB) strings
        try:
            return from_shapely(shapely.from_wkb(obj))
        except (TypeError, shapely.errors.GEOSException):
            ...

        # JSON/GEOJSON dictionary
        return from_shapely(obj.map(self._coerce_element))  # type: ignore[operator]

    def _coerce_element(self, element: Any) -> Any:
        try:
            return shapely.geometry.shape(element)
        except (
            AttributeError,
            TypeError,
            shapely.errors.GeometryTypeError,
            shapely.errors.GEOSException,
        ):
            return np.nan

    def _coerce_crs(self, value: GeoPandasObject) -> GeoPandasObject:
        if self.crs is not None:
            if value.crs is None:
                # Allow assignment of CRS if currently
                # null and a non-null value is designated.
                # This will only work in the context of
                # geopandas because assinging geometry
                # CRS to a pandas dataframe isn't supported.
                value.crs = self.crs
            elif isinstance(value, gpd.GeoSeries) and self.crs != value.crs:
                value = value.to_crs(self.crs)  # type: ignore[operator]
            elif isinstance(value, gpd.GeoDataFrame) and any(
                self.crs != value[col].crs for col in value.columns
            ):
                for col in value.columns:
                    if self.crs != value[col].crs:
                        value[col] = value[col].to_crs(self.crs)
        return value

    def coerce(self, data_container: GeoPandasObject) -> GeoPandasObject:
        """Coerce data container to the specified data type."""
        # pylint: disable=import-outside-toplevel
        from pandera.backends.pandas import error_formatters

        orig_isna = data_container.isna()

        # Copy so we don't directly modify container due
        # to CRS re-projection, etc.)
        data_container = data_container.copy()

        # Coerce container data
        coerced_data = self._coerce_values(data_container)

        # Coerce container type
        if isinstance(coerced_data, (GeometryArray, pd.DataFrame)):
            if isinstance(data_container, (pd.Series, gpd.GeoSeries)):
                coerced_data = gpd.GeoSeries(coerced_data)
            else:
                coerced_data = gpd.GeoDataFrame(coerced_data)

        failed_selector = coerced_data.isna() & ~orig_isna

        if np.any(failed_selector.any()):
            failure_cases = coerced_data[failed_selector]
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=error_formatters.reshape_failure_cases(
                    failure_cases, ignore_na=False
                ),
            )
        coerced = self._coerce_crs(coerced_data)
        return coerced

    def check(  # type: ignore
        self,
        pandera_dtype: pandas_engine.DataType,
        data_container: Optional[GeoPandasObject] = None,
    ) -> Union[bool, Iterable[bool]]:
        """Check data container to the specified data type."""
        # Type check
        if not super().check(pandera_dtype, data_container):
            if data_container is None:
                return False
            else:
                return np.full_like(data_container, False, dtype=bool)
        if self.crs != pandera_dtype.crs and data_container is None:  # type: ignore[attr-defined]
            return False

        # CRS check extends into container
        if self.crs is not None:
            if (
                isinstance(data_container, gpd.GeoSeries)
                and data_container.crs != self.crs
            ):
                # GeoSeries
                raise TypeError(
                    f"CRS mismatch; actual {str(data_container.crs)}, expected {str(self.crs)}"
                )
            if isinstance(data_container, gpd.GeoDataFrame):
                # GeoDataFrame
                col_results = []
                for col in data_container.columns:
                    if data_container[col].crs != self.crs:
                        col_err = f"CRS mismatch on column {col}; actual {str(data_container[col].crs)}, expected {str(self.crs)}"
                        col_results.append(col_err)
                if col_results:
                    raise TypeError("\n".join(col_results))

        return np.full_like(data_container, True, dtype=bool)

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, type(self)):
            return obj.crs == self.crs
        return super().__eq__(obj)

    def __str__(self) -> str:
        return "geometry"
