"""Pandera type annotations for GeoPandas."""
import functools
from typing import (  # type: ignore[attr-defined]
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    _type_check,
)

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

import pandas as pd

from pandera.engines import PYDANTIC_V2
from pandera.errors import SchemaError, SchemaInitError
from pandera.typing.common import (
    DataFrameBase,
    DataFrameModel,
    SeriesBase,
)
from pandera.typing.formats import Formats

if PYDANTIC_V2:
    from pydantic_core import core_schema
    from pydantic import GetCoreSchemaHandler

try:
    import geopandas as gpd

    try:
        from typing import _GenericAlias  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        _GenericAlias = None

    GEOPANDAS_INSTALLED = True
except ImportError:  # pragma: no cover
    GEOPANDAS_INSTALLED = False


if GEOPANDAS_INSTALLED:
    # pylint: disable=import-outside-toplevel,ungrouped-imports
    from pandera.engines.pandas_engine import Geometry

    # pylint:disable=invalid-name
    if TYPE_CHECKING:
        T = TypeVar("T")  # pragma: no cover
    else:
        T = DataFrameModel

    # pylint:disable=too-few-public-methods
    class GeoSeries(SeriesBase, gpd.GeoSeries, Generic[T]):
        """
        Representation of geopandas.GeoSeries, only used for type annotation.
        """

        default_dtype = Geometry

    class GeoDataFrame(DataFrameBase, gpd.GeoDataFrame, Generic[T]):
        """
        A generic type for geopandas.GeoDataFrame.
        """

        if hasattr(gpd.GeoDataFrame, "__class_getitem__") and _GenericAlias:

            def __class_getitem__(cls, item):
                """Define this to override the patch that pyspark.pandas performs on pandas.
                https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
                """
                _type_check(item, "Parameters to generic types must be types.")
                return _GenericAlias(cls, item)

        @classmethod
        def _activate_geometry(cls, obj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            try:
                # Test for presence of active geometry attribute
                _ = obj.geometry
            except AttributeError:
                # Assign active geometry
                geometry_cols = obj.select_dtypes(include=["geometry"]).columns
                if "geometry" in geometry_cols:
                    # Standard column name
                    return obj.set_geometry("geometry")

            # Return as-is, either because an active geometry has
            # already been assigned, or because we only have
            # geometry columns not named "geometry", meaning the
            # user will have to set manually. This follows
            # GeoPandas' GeoDataFrame constructor behavior.
            return obj

        @classmethod
        def from_format(cls, obj: Any, config) -> gpd.GeoDataFrame:
            """
            Converts serialized data from a specific format
            specified in the :py:class:`pandera.api.pandas.model.DataFrameModel` config options
            ``from_format`` and ``from_format_kwargs``.

            :param obj: object representing a serialized dataframe.
            :param config: dataframe model configuration object.
            """
            if config.from_format is None:
                if not isinstance(obj, gpd.GeoDataFrame):
                    try:
                        obj = gpd.GeoDataFrame(obj)

                        # It's possible that geometries in non-standard
                        # formats (geojson dicts, wkt, etc.) may not have
                        # been properly coerced to GeoSeries through the
                        # basic GeoDataFrame constructor, but attempt to
                        # activate geometry anyway. Otherwise, if we can't
                        # activate geometry now then we'll try again post-
                        # pydantic validation.
                        obj = cls._activate_geometry(obj)
                    except Exception as exc:
                        raise ValueError(
                            f"Expected gpd.GeoDataFrame, found {type(obj)}"
                        ) from exc
                return obj

            if callable(config.from_format):
                reader = config.from_format
            else:
                reader = {
                    Formats.dict: gpd.GeoDataFrame,
                }[Formats(config.from_format)]
            return reader(obj, **(config.from_format_kwargs or {}))  # type: ignore

        @classmethod
        def to_format(cls, data: gpd.GeoDataFrame, config) -> Any:
            """
            Converts a geodataframe to the format specified in the
            :py:class:`pandera.api.pandas.model.DataFrameModel` config options ``to_format``
            and ``to_format_kwargs``.

            :param data: convert this data to the specified format
            :param config: :py:cl
            """
            if config.to_format is None:
                return data

            if callable(config.to_format):
                writer = functools.partial(config.to_format, data)
                if callable(config.to_format_buffer):
                    buffer = config.to_format_buffer()
                elif config.to_format_buffer is None:
                    buffer = None
                else:  # pragma: no cover
                    raise TypeError(
                        "to_format_buffer must be Callable or None, found "
                        f"{config.to_format_buffer}"
                    )
            else:
                writer, buffer = {  # type: ignore[assignment]
                    Formats.dict: (data.to_dict, None),
                }[Formats(config.to_format)]

            args = [] if buffer is None else [buffer]
            out = writer(*args, **(config.to_format_kwargs or {}))  # type: ignore
            if buffer is None:
                return out
            elif buffer.closed:
                raise IOError(
                    f"pandas=={pd.__version__} closed the buffer automatically "
                    f"using the serialization method {writer}. Use a later "
                    "version of pandas or use a different the serialization "
                    "format."
                )
            buffer.seek(0)
            return buffer

        @classmethod
        def _get_schema_model(cls, field):
            if not field.sub_fields:
                raise TypeError(
                    "Expected a typed pandera.typing.GeoDataFrame,"
                    " e.g. GeoDataFrame[Schema]"
                )
            schema_model = field.sub_fields[0].type_
            return schema_model

        if PYDANTIC_V2:

            @classmethod
            def __get_pydantic_core_schema__(
                cls, _source_type: Any, _handler: GetCoreSchemaHandler
            ) -> core_schema.CoreSchema:
                schema_model = get_args(_source_type)[0]
                return core_schema.no_info_plain_validator_function(
                    functools.partial(
                        cls.pydantic_validate,
                        schema_model=schema_model,
                    ),
                )

        else:

            @classmethod
            def __get_validators__(cls):
                yield cls._pydantic_validate

        @classmethod
        def pydantic_validate(cls, obj: Any, schema_model) -> gpd.GeoDataFrame:
            """
            Verify that the input can be converted into a pandas dataframe that
            meets all schema requirements.

            This is for pydantic >= v2
            """
            try:
                schema = schema_model.to_schema()
            except SchemaInitError as exc:
                raise ValueError(
                    f"Cannot use {cls.__name__} as a pydantic type as its "
                    "DataFrameModel cannot be converted to a DataFrameSchema.\n"
                    f"Please revisit the model to address the following errors:"
                    f"\n{exc}"
                ) from exc
            data = cls.from_format(obj, schema_model.__config__)

            try:
                valid_data = schema.validate(data)
                valid_data = cls._activate_geometry(valid_data)
            except SchemaError as exc:
                raise ValueError(str(exc)) from exc

            return cls.to_format(valid_data, schema_model.__config__)

        @classmethod
        def _pydantic_validate(cls, obj: Any, field) -> gpd.GeoDataFrame:
            """
            Verify that the input can be converted into a geopandas geodataframe that
            meets all schema requirements.

            This is for pydantic < v1
            """
            schema_model = cls._get_schema_model(field)
            return cls.pydantic_validate(obj, schema_model)
