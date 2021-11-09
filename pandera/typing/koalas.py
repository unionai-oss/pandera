"""Pandera type annotations for Dask."""

from typing import TYPE_CHECKING, Generic, TypeVar

from .common import DataFrameBase, IndexBase, SeriesBase
from .pandas import GenericDtype, Schema, _GenericAlias

try:
    import databricks.koalas as ks

    KOALAS_INSTALLED = True
except ImportError:
    KOALAS_INSTALLED = False


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


if KOALAS_INSTALLED:

    # pylint: disable=too-few-public-methods
    class DataFrame(DataFrameBase, ks.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

        def __class_getitem__(cls, item):
            """Define this to override's koalas generic type."""
            return _GenericAlias(cls, item)

    # pylint:disable=too-few-public-methods
    class Series(SeriesBase, ks.Series, Generic[GenericDtype]):
        """Representation of pandas.Series, only used for type annotation.

        *new in 0.8.0*
        """

        def __class_getitem__(cls, item):
            """Define this to override koalas generic type"""
            return _GenericAlias(cls, item)

    # pylint:disable=too-few-public-methods
    class Index(IndexBase, ks.Index, Generic[GenericDtype]):
        """Representation of pandas.Index, only used for type annotation.

        *new in 0.8.0*
        """
