"""Pandera type annotations for Dask."""

try:
    import pyspark.sql as ps

    PYSPARK_SQL_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_SQL_INSTALLED = False

if PYSPARK_SQL_INSTALLED:
    from typing import TYPE_CHECKING, Generic, TypeVar

    from pandera.typing.common import ColumnBase, DataFrameBase, GenericDtype
    from pandera.typing.pandas import DataFrameModel, _GenericAlias

    # pylint:disable=invalid-name
    if TYPE_CHECKING:
        T = TypeVar("T")  # pragma: no cover
    else:
        T = DataFrameModel

    if PYSPARK_SQL_INSTALLED:
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

        class Column(ColumnBase, ps.Column, Generic[GenericDtype]):  # type: ignore [misc]  # noqa
            """Representation of pyspark.sql.Column, only used for type annotation."""
