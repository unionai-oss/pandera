"""Pandera type annotations for Polars."""

from typing import TYPE_CHECKING, Generic, TypeVar

from packaging import version

from pandera.typing.common import DataFrameBase, DataFrameModel, SeriesBase

try:
    import polars as pl

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False


def polars_version():
    """Return the modin version."""
    return version.parse(pl.__version__)


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if POLARS_INSTALLED:
    # pylint: disable=too-few-public-methods
    class LazyFrame(DataFrameBase, pl.LazyFrame, Generic[T]):
        """
        Pandera generic for pl.LazyFrame, only used for type annotation.

        *new in 0.19.0*
        """

    class DataFrame(DataFrameBase, pl.DataFrame, Generic[T]):
        """
        Pandera generic for pl.LazyFrame, only used for type annotation.

        *new in 0.19.0*
        """

    # pylint: disable=too-few-public-methods
    class Series(SeriesBase, pl.Series, Generic[T]):
        """
        Pandera generic for pl.Series, only used for type annotation.

        *new in 0.19.0*
        """
