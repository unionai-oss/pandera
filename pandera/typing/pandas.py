"""Typing definitions and helpers."""

import functools
import io
from typing import (  # type: ignore[attr-defined]
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
    _type_check,
)

import numpy as np
import pandas as pd

from pandera import dtypes
from pandera.config import config_context
from pandera.engines import PYDANTIC_V2, numpy_engine, pandas_engine
from pandera.errors import SchemaError, SchemaInitError
from pandera.typing.common import (
    DataFrameBase,
    DataFrameModel,
    IndexBase,
    SeriesBase,
)
from pandera.typing.common import (
    GenericDtype as _CommonGenericDtype,
)
from pandera.typing.formats import Formats

try:
    from typing import _GenericAlias  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _GenericAlias = None


if PYDANTIC_V2:
    import pydantic_core
    from packaging import version
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema


Bool = dtypes.Bool  #: ``"bool"`` numpy dtype
Date = dtypes.Date  #: ``datetime.date`` object dtype
DateTime = dtypes.DateTime  #: ``"datetime64[ns]"`` numpy dtype
Decimal = dtypes.Decimal  #: ``decimal.Decimal`` object dtype
Timedelta = dtypes.Timedelta  #: ``"timedelta64[ns]"`` numpy dtype
Category = dtypes.Category  #: pandas ``"categorical"`` datatype
Float = dtypes.Float  #: ``"float"`` numpy dtype
Float16 = dtypes.Float16  #: ``"float16"`` numpy dtype
Float32 = dtypes.Float32  #: ``"float32"`` numpy dtype
Float64 = dtypes.Float64  #: ``"float64"`` numpy dtype
Int = dtypes.Int  #: ``"int"`` numpy dtype
Int8 = dtypes.Int8  #: ``"int8"`` numpy dtype
Int16 = dtypes.Int16  #: ``"int16"`` numpy dtype
Int32 = dtypes.Int32  #: ``"int32"`` numpy dtype
Int64 = dtypes.Int64  #: ``"int64"`` numpy dtype
UInt8 = dtypes.UInt8  #: ``"uint8"`` numpy dtype
UInt16 = dtypes.UInt16  #: ``"uint16"`` numpy dtype
UInt32 = dtypes.UInt32  #: ``"uint32"`` numpy dtype
UInt64 = dtypes.UInt64  #: ``"uint64"`` numpy dtype
INT8 = pandas_engine.INT8  #: ``"Int8"`` pandas dtype:: pandas 0.24.0+
INT16 = pandas_engine.INT16  #: ``"Int16"`` pandas dtype: pandas 0.24.0+
INT32 = pandas_engine.INT32  #: ``"Int32"`` pandas dtype: pandas 0.24.0+
INT64 = pandas_engine.INT64  #: ``"Int64"`` pandas dtype: pandas 0.24.0+
UINT8 = pandas_engine.UINT8  #: ``"UInt8"`` pandas dtype:: pandas 0.24.0+
UINT16 = pandas_engine.UINT16  #: ``"UInt16"`` pandas dtype: pandas 0.24.0+
UINT32 = pandas_engine.UINT32  #: ``"UInt32"`` pandas dtype: pandas 0.24.0+
UINT64 = pandas_engine.UINT64  #: ``"UInt64"`` pandas dtype: pandas 0.24.0+
Object = numpy_engine.Object  #: ``"object"`` numpy dtype
String = dtypes.String  #: ``"str"`` numpy dtype
#: ``"string"`` pandas dtypes: pandas 1.0.0+. For <1.0.0, this enum will
#: fall back on the str-as-object-array representation.
STRING = pandas_engine.STRING  #: ``"str"`` numpy dtype
BOOL = pandas_engine.BOOL  #: ``"str"`` numpy dtype


GenericDtype = TypeVar(  # type: ignore
    "GenericDtype",
    bound=Union[
        bool,
        int,
        str,
        float,
        Bool,
        Date,
        DateTime,
        Decimal,
        Timedelta,
        Category,
        Float,
        Float16,
        Float32,
        Float64,
        Int,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        pd.core.dtypes.base.ExtensionDtype,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        Object,
        String,
        STRING,
    ],
)


class Index(IndexBase, pd.Index, Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """


class Series(SeriesBase, pd.Series, Generic[GenericDtype]):  # type: ignore
    """Representation of pandas.Series, only used for type annotation.

    *new in 0.5.0*
    """

    def __class_getitem__(cls, item):
        """Define this to override the patch that pyspark.pandas performs on pandas.
        https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
        """
        _type_check(item, "Parameters to generic types must be types.")
        return _GenericAlias(cls, item)


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


class DataFrame(DataFrameBase, pd.DataFrame, Generic[T]):
    """
    A generic type for pandas.DataFrame.

    *new in 0.5.0*
    """

    def __class_getitem__(cls, item):
        """Define this to override the patch that pyspark.pandas performs on pandas.
        https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
        """
        _type_check(item, "Parameters to generic types must be types.")
        return _GenericAlias(cls, item)

    @classmethod
    def from_format(cls, obj: Any, config) -> pd.DataFrame:
        """
        Converts serialized data from a specific format
        specified in the :py:class:`pandera.api.pandas.model.DataFrameModel` config options
        ``from_format`` and ``from_format_kwargs``.

        :param obj: object representing a serialized dataframe.
        :param config: dataframe model configuration object.
        """
        if config.from_format is None:
            if not isinstance(obj, pd.DataFrame):
                try:
                    obj = pd.DataFrame(obj)
                except Exception as exc:
                    raise ValueError(
                        f"Expected pd.DataFrame, found {type(obj)}"
                    ) from exc
            return obj

        if callable(config.from_format):
            reader = config.from_format
        else:
            reader = {
                Formats.dict: pd.DataFrame.from_dict,
                Formats.csv: pd.read_csv,
                Formats.json: pd.read_json,
                Formats.feather: pd.read_feather,
                Formats.parquet: pd.read_parquet,
                Formats.pickle: pd.read_pickle,
                Formats.json_normalize: pd.json_normalize,
            }[Formats(config.from_format)]

        return reader(obj, **(config.from_format_kwargs or {}))  # type: ignore

    @classmethod
    def to_format(cls, data: pd.DataFrame, config) -> Any:
        """
        Converts a dataframe to the format specified in the
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
                Formats.csv: (data.to_csv, None),
                Formats.json: (data.to_json, None),
                Formats.feather: (data.to_feather, io.BytesIO()),
                Formats.parquet: (data.to_parquet, io.BytesIO()),
                Formats.pickle: (data.to_pickle, io.BytesIO()),
            }[Formats(config.to_format)]

        args = [] if buffer is None else [buffer]
        out = writer(*args, **(config.to_format_kwargs or {}))  # type: ignore
        if buffer is None:
            return out
        elif buffer.closed:
            raise OSError(
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
                "Expected a typed pandera.typing.DataFrame,"
                " e.g. DataFrame[Schema]"
            )
        schema_model = field.sub_fields[0].type_
        return schema_model

    if PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            # prevent validation in __setattr__ function in DataFrameBase class
            with config_context(validation_enabled=False):
                schema_model = _source_type().__orig_class__.__args__[0]

            function = functools.partial(
                cls.pydantic_validate,
                schema_model=schema_model,
            )

            if version.parse(pydantic_core.__version__).release < (
                2,
                30,
                0,
            ) or isinstance(schema_model, TypeVar):
                return core_schema.no_info_plain_validator_function(function)
            else:
                schema = schema_model.to_schema()
                schema_json_columns = schema_model.to_json_schema()[
                    "properties"
                ]
                type_map = {
                    "string": core_schema.str_schema(),
                    "integer": core_schema.int_schema(),
                    "number": core_schema.float_schema(),
                    "boolean": core_schema.bool_schema(),
                    "datetime": core_schema.datetime_schema(),
                    "duration": core_schema.timedelta_schema(),
                    "any": core_schema.any_schema(),
                }

                json_schema_input_schema = core_schema.list_schema(
                    core_schema.typed_dict_schema(
                        {
                            key: core_schema.typed_dict_field(
                                type_map[
                                    schema_json_columns[key]["items"]["type"]
                                ]
                            )
                            for key in schema.columns.keys()
                        },
                    )
                )
                # json schema input schema is only available in
                # pydantic_core >=2.30.0
                return core_schema.no_info_plain_validator_function(
                    function,
                    json_schema_input_schema=json_schema_input_schema,
                    serialization=core_schema.plain_serializer_function_ser_schema(
                        function=lambda df: df,
                        info_arg=False,
                        return_schema=json_schema_input_schema,
                    ),
                )

    else:

        @classmethod
        def __get_validators__(cls):
            yield cls._pydantic_validate

    @classmethod
    def pydantic_validate(cls, obj: Any, schema_model) -> pd.DataFrame:
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
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc

        return cls.to_format(valid_data, schema_model.__config__)

    @classmethod
    def _pydantic_validate(cls, obj: Any, field) -> pd.DataFrame:
        """
        Verify that the input can be converted into a pandas dataframe that
        meets all schema requirements.

        This is for pydantic < v1
        """
        schema_model = cls._get_schema_model(field)
        return cls.pydantic_validate(obj, schema_model)

    @staticmethod
    def from_records(  # type: ignore
        schema: type[T],
        data: Union[  # type: ignore
            np.ndarray, list[tuple[Any, ...]], dict[Any, Any], pd.DataFrame
        ],
        **kwargs,
    ) -> "DataFrame[T]":
        """
        Convert structured or record ndarray to pandera-validated DataFrame.

        Creates a DataFrame object from a structured ndarray, sequence of tuples
        or dicts, or DataFrame.

        See :doc:`pandas:reference/api/pandas.DataFrame.from_records` for
        more details.
        """
        schema = schema.to_schema()  # type: ignore[attr-defined]
        schema_index = schema.index.names if schema.index is not None else None
        if "index" not in kwargs:
            kwargs["index"] = schema_index
        data_df = pd.DataFrame.from_records(data=data, **kwargs)
        return DataFrame[schema](  # type: ignore
            # set the column order according to schema
            data_df[[c for c in schema.columns if c in data_df.columns]]
        )
