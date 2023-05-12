"""Pandera type annotations for Dask."""

from typing import TYPE_CHECKING, Generic, TypeVar

from packaging import version

from pandera.typing.common import DataFrameBase, IndexBase, SeriesBase
from pandera.typing.pandas import DataFrameModel, GenericDtype

try:
    import modin
    import modin.pandas as mpd

    MODIN_INSTALLED = True
except ImportError:
    MODIN_INSTALLED = False


def modin_version():
    """Return the modin version."""
    return version.parse(modin.__version__)


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


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
