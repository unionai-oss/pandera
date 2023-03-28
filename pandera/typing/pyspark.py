"""Pandera type annotations for Dask."""

from typing import TYPE_CHECKING, Generic, TypeVar

from pandera.typing.common import DataFrameBase, IndexBase, SeriesBase
from pandera.typing.pandas import GenericDtype, DataFrameModel, _GenericAlias

try:
    import pyspark.pandas as ps

    PYSPARK_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_INSTALLED = False


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if PYSPARK_INSTALLED:

    # pylint: disable=too-few-public-methods,arguments-renamed
    class DataFrame(DataFrameBase, ps.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

        def __class_getitem__(cls, item):
            """Define this to override's pyspark.pandas generic type."""
            return _GenericAlias(cls, item)
