"""Pandera type annotations for Dask."""

import inspect
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .common import DataFrameBase, IndexBase, SeriesBase
from .pandas import GenericDtype, Schema

try:
    import dask.dataframe as dd

    DASK_INSTALLED = True
except ImportError:
    DASK_INSTALLED = False


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


if DASK_INSTALLED:

    # pylint: disable=too-few-public-methods
    class DataFrame(DataFrameBase, dd.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

        def __setattr__(self, name: str, value: Any) -> None:
            object.__setattr__(self, name, value)
            if name == "__orig_class__":
                class_args = getattr(self.__orig_class__, "__args__", None)
                if any(
                    x.__name__ == "SchemaModel"
                    for x in inspect.getmro(class_args[0])
                ):
                    schema_model = value.__args__[0]

                # prevent the double validation problem by preventing checks
                # for dataframes with a defined pandera.schema
                if (
                    self.pandera.schema is None
                    or self.pandera.schema != schema_model.to_schema()
                ):
                    # pylint: disable=self-cls-assignment
                    self.__dict__ = schema_model.validate(self).__dict__
                    self.pandera.add_schema(schema_model.to_schema())

    # pylint:disable=too-few-public-methods
    class Series(SeriesBase, dd.Series, Generic[GenericDtype]):  # type: ignore
        """Representation of pandas.Series, only used for type annotation.

        *new in 0.8.0*
        """

    # pylint:disable=too-few-public-methods
    class Index(IndexBase, dd.Index, Generic[GenericDtype]):
        """Representation of pandas.Index, only used for type annotation.

        *new in 0.8.0*
        """
