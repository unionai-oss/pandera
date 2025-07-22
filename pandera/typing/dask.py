"""Pandera type annotations for Dask."""

from typing import TYPE_CHECKING, Generic, TypeVar

from pandera.typing.common import DataFrameBase, IndexBase, SeriesBase
from pandera.typing.pandas import DataFrameModel, GenericDtype

try:
    import dask.dataframe as dd

    DASK_INSTALLED = True
except ImportError:
    DASK_INSTALLED = False


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if DASK_INSTALLED:

    class DataFrame(DataFrameBase, dd.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

    class Series(SeriesBase, dd.Series, Generic[GenericDtype]):  # type: ignore
        """Representation of pandas.Series, only used for type annotation.

        *new in 0.8.0*
        """

    class Index(IndexBase, dd.Index, Generic[GenericDtype]):
        """Representation of pandas.Index, only used for type annotation.

        *new in 0.8.0*
        """
