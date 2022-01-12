"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pandas as pd

from ..errors import SchemaError, SchemaInitError
from .common import DataFrameBase, GenericDtype, IndexBase, Schema, SeriesBase

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
    def _pydantic_validate(
        cls, df: pd.DataFrame, field: ModelField
    ) -> pd.DataFrame:
        """Verify that the input is a pandas dataframe that meets all
        schema requirements."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

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

        try:
            return schema.validate(df)
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc
