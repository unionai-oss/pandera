"""Pandera type annotations for Dask."""

from typing import TYPE_CHECKING, Generic, TypeVar

from .common import DataFrameBase, IndexBase, SeriesBase
from .pandas import GenericDtype, Schema

try:
    import modin.pandas as mpd

    MODIN_INSTALLED = True
except ImportError:
    MODIN_INSTALLED = False


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


if MODIN_INSTALLED:

    # pylint: disable=too-few-public-methods
    class DataFrame(DataFrameBase, mpd.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

    # pylint:disable=too-few-public-methods,abstract-method
    class Series(SeriesBase, mpd.Series, Generic[GenericDtype]):
        """Representation of pandas.Series, only used for type annotation.

        *new in 0.8.0*
        """

    # pylint:disable=too-few-public-methods,abstract-method
    class Index(IndexBase, mpd.Index, Generic[GenericDtype]):
        """Representation of pandas.Index, only used for type annotation.

        *new in 0.8.0*
        """
