"""Pandera type annotations for Modin."""

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


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if MODIN_INSTALLED:

    class DataFrame(DataFrameBase, mpd.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

    class Series(SeriesBase, mpd.Series, Generic[GenericDtype]):
        """Representation of pandas.Series, only used for type annotation.

        *new in 0.8.0*
        """

    class Index(IndexBase, mpd.Index, Generic[GenericDtype]):
        """Representation of pandas.Index, only used for type annotation.

        *new in 0.8.0*
        """
