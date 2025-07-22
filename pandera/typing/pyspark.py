"""Pandera type annotations for Pyspark Pandas."""

from typing import TYPE_CHECKING, Generic, TypeVar

from pandera.typing.common import (
    DataFrameBase,
    GenericDtype,
    IndexBase,
    SeriesBase,
)
from pandera.typing.pandas import DataFrameModel, _GenericAlias

try:
    import pyspark.pandas as ps

    PYSPARK_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_INSTALLED = False


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if PYSPARK_INSTALLED:

    class DataFrame(DataFrameBase, ps.DataFrame, Generic[T]):
        """
        Representation of dask.dataframe.DataFrame, only used for type
        annotation.

        *new in 0.8.0*
        """

        def __class_getitem__(cls, item):
            """Define this to override's pyspark.pandas generic type."""
            return _GenericAlias(cls, item)

    class Series(SeriesBase, ps.Series, Generic[GenericDtype]):  # type: ignore [misc]  # noqa
        """Representation of pandas.Series, only used for type annotation.

        *new in 0.8.0*
        """

        def __class_getitem__(cls, item):
            """Define this to override pyspark.pandas generic type"""
            return _GenericAlias(cls, item)

    class Index(IndexBase, ps.Index, Generic[GenericDtype]):
        """Representation of pandas.Index, only used for type annotation.

        *new in 0.8.0*
        """
