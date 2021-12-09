"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
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
        yield cls._pydantic_validate

    @classmethod
    def from_pre_format(cls, obj: Any, config) -> pd.DataFrame:
        if config.pre_format is None:
            return (
                pd.DataFrame(obj) if not isinstance(obj, pd.DataFrame) else obj
            )

        if Formats(config.pre_format) == Formats.csv:
            data = pd.read_csv(obj, **(config.pre_format_options or {}))

        return data

    @classmethod
    def to_post_format(cls, data: pd.DataFrame, config) -> Any:
        if config.post_format is None:
            return data

        if Formats(config.post_format) == Formats.dict:
            data = data.to_dict(**(config.post_format_options or {}))
        return data

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
    def _pydantic_validate(cls, obj: Any, field: ModelField) -> pd.DataFrame:
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
