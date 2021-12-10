"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
import io
from typing import TYPE_CHECKING, Any, Dict, Generic, TypeVar, Union

import pandas as pd

from ..errors import SchemaError, SchemaInitError
from .common import DataFrameBase, GenericDtype, IndexBase, Schema, SeriesBase
from .formats import Formats

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
            """Define this to override the patch that koalas performs on pandas.
            https://github.com/databricks/koalas/blob/master/databricks/koalas/__init__.py#L207-L223
            """
            return _GenericAlias(cls, item)


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


# pylint:disable=too-few-public-methods
class DataFrame(DataFrameBase, pd.DataFrame, Generic[T]):
    """
    A generic type for pandas.DataFrame.

    *new in 0.5.0*
    """

    if hasattr(pd.DataFrame, "__class_getitem__") and _GenericAlias:

        def __class_getitem__(cls, item):
            """Define this to override the patch that koalas performs on pandas.
            https://github.com/databricks/koalas/blob/master/databricks/koalas/__init__.py#L207-L223
            """
            return _GenericAlias(cls, item)

    @classmethod
    def __get_validators__(cls):
        yield cls.pydantic_validate

    @classmethod
    def from_pre_format(cls, obj: Any, config) -> pd.DataFrame:
        if config.pre_format is None:
            if not isinstance(obj, pd.DataFrame):
                raise ValueError(f"Expected pd.DataFrame, found {type(obj)}")
            return obj

        reader = {
            Formats.dict: pd.DataFrame,
            Formats.csv: pd.read_csv,
            Formats.json: pd.read_json,
            Formats.feather: pd.read_feather,
            Formats.parquet: pd.read_parquet,
            Formats.pickle: pd.read_pickle,
        }[Formats(config.pre_format)]

        return reader(obj, **(config.pre_format_options or {}))

    @classmethod
    def to_post_format(cls, data: pd.DataFrame, config) -> Any:
        if config.post_format is None:
            return data

        writer, buffer = {
            Formats.dict: (data.to_dict, None),
            Formats.csv: (data.to_csv, None),
            Formats.json: (data.to_json, None),
            Formats.feather: (data.to_feather, io.BytesIO()),
            Formats.parquet: (data.to_parquet, io.BytesIO()),
            Formats.pickle: (data.to_pickle, io.BytesIO()),
        }[Formats(config.post_format)]

        args = [] if buffer is None else [buffer]
        out = writer(*args, **(config.post_format_options or {}))
        if buffer is None:
            return out
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
                "SchemaModel cannot be converted to a DataFrameSchema.\n"
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
        data = cls.from_pre_format(obj, schema_model.__config__)

        try:
            valid_data = schema.validate(data)
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc

        return cls.to_post_format(valid_data, schema_model.__config__)
