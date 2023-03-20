"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
import functools
import io
from typing import (  # type: ignore[attr-defined]
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Tuple,
    TypeVar,
    Union,
    _type_check,
)

import numpy as np
import pandas as pd

from pandera.errors import SchemaError, SchemaInitError
from pandera.typing.common import (
    DataFrameBase,
    GenericDtype,
    IndexBase,
    DataFrameModel,
    SeriesBase,
)
from pandera.typing.formats import Formats

try:
    from typing import _GenericAlias  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _GenericAlias = None


try:
    from pydantic.fields import ModelField
except ImportError:
    ModelField = Any  # type: ignore


# pylint:disable=too-few-public-methods
class Index(IndexBase, pd.Index, Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """


# pylint:disable=too-few-public-methods
class Series(SeriesBase, pd.Series, Generic[GenericDtype]):  # type: ignore
    """Representation of pandas.Series, only used for type annotation.

    *new in 0.5.0*
    """

    if hasattr(pd.Series, "__class_getitem__") and _GenericAlias:

        def __class_getitem__(cls, item):
            """Define this to override the patch that pyspark.pandas performs on pandas.
            https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
            """
            _type_check(item, "Parameters to generic types must be types.")
            return _GenericAlias(cls, item)


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


# pylint:disable=too-few-public-methods
class DataFrame(DataFrameBase, pd.DataFrame, Generic[T]):
    """
    A generic type for pandas.DataFrame.

    *new in 0.5.0*
    """

    if hasattr(pd.DataFrame, "__class_getitem__") and _GenericAlias:

        def __class_getitem__(cls, item):
            """Define this to override the patch that pyspark.pandas performs on pandas.
            https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
            """
            _type_check(item, "Parameters to generic types must be types.")
            return _GenericAlias(cls, item)

    @classmethod
    def __get_validators__(cls):
        yield cls.pydantic_validate

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
                Formats.dict: pd.DataFrame,
                Formats.csv: pd.read_csv,
                Formats.json: pd.read_json,
                Formats.feather: pd.read_feather,
                Formats.parquet: pd.read_parquet,
                Formats.pickle: pd.read_pickle,
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
            raise IOError(
                f"pandas=={pd.__version__} closed the buffer automatically "
                f"using the serialization method {writer}. Use a later "
                "version of pandas or use a different the serialization "
                "format."
            )
        buffer.seek(0)
        return buffer

    @classmethod
    def _get_schema(cls, field: ModelField):
        if not field.sub_fields:
            raise TypeError(
                "Expected a typed pandera.typing.DataFrame,"
                " e.g. DataFrame[Schema]"
            )
        schema_model = field.sub_fields[0].type_
        try:
            schema = schema_model.to_schema()
        except SchemaInitError as exc:
            raise ValueError(
                f"Cannot use {cls.__name__} as a pydantic type as its "
                "DataFrameModel cannot be converted to a DataFrameSchema.\n"
                f"Please revisit the model to address the following errors:"
                f"\n{exc}"
            ) from exc
        return schema_model, schema

    @classmethod
    def pydantic_validate(cls, obj: Any, field: ModelField) -> pd.DataFrame:
        """
        Verify that the input can be converted into a pandas dataframe that
        meets all schema requirements.
        """
        schema_model, schema = cls._get_schema(field)
        data = cls.from_format(obj, schema_model.__config__)

        try:
            valid_data = schema.validate(data)
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc

        return cls.to_format(valid_data, schema_model.__config__)

    @staticmethod
    def from_records(  # type: ignore
        schema: T,
        data: Union[  # type: ignore
            np.ndarray, List[Tuple[Any, ...]], Dict[Any, Any], pd.DataFrame
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
        return DataFrame[T](
            pd.DataFrame.from_records(data=data, **kwargs,)[
                schema.columns.keys()
            ]  # set the column order according to schema
        )
