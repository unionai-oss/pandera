"""Pandera type annotations for Dask."""
from pandera.typing.common import DataFrameBase, GenericDtype
from pandera.typing.pandas import DataFrameModel, _GenericAlias
from typing import Union, Optional, Type, TypeVar

try:
    import pyspark.sql as ps

    PYSPARK_SQL_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_SQL_INSTALLED = False

if PYSPARK_SQL_INSTALLED:
    from pandera.engines import pyspark_engine

    PYSPARK_STRING = pyspark_engine.String
    PYSPARK_INT = pyspark_engine.Int
    PYSPARK_LONGINT = pyspark_engine.BigInt
    PYSPARK_SHORTINT = pyspark_engine.ShortInt
    PYSPARK_BYTEINT = pyspark_engine.ByteInt
    PYSPARK_DOUBLE = pyspark_engine.Double
    PYSPARK_FLOAT = pyspark_engine.Float
    PYSPARK_DECIMAL = pyspark_engine.Decimal
    PYSPARK_DATE = pyspark_engine.Date
    PYSPARK_TIMESTAMP = pyspark_engine.Timestamp
    PYSPARK_BINARY = pyspark_engine.Binary


    PysparkDType = TypeVar(  # type: ignore
        "PysparkDType",
        bound=Union[
                    PYSPARK_STRING,
                    PYSPARK_INT,
                    PYSPARK_LONGINT,
                    PYSPARK_SHORTINT,
                    PYSPARK_BYTEINT,
                    PYSPARK_FLOAT,
                    PYSPARK_DECIMAL,
                    PYSPARK_DATE,
                    PYSPARK_TIMESTAMP,
                    PYSPARK_BINARY,
        ],
    )
    from typing import TYPE_CHECKING, Generic, TypeVar

    # pylint:disable=invalid-name
    if TYPE_CHECKING:
        T = TypeVar("T")  # pragma: no cover
    else:
        T = DataFrameModel

    if PYSPARK_SQL_INSTALLED:
        # pylint: disable=too-few-public-methods,arguments-renamed
        class ColumnBase(Generic[PysparkDType]):
            """Representation of pandas.Index, only used for type annotation.

            *new in 0.5.0*
            """

            default_dtype: Optional[Type] = None

            def __get__(self, instance: object, owner: Type) -> str:  # pragma: no cover
                raise AttributeError("column should resolve to pyspark.sql.Column-s")


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
