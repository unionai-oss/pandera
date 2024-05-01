"""Pandera type annotations for Pyspark."""

from typing import TypeVar, Union

from pandera.typing.common import DataFrameBase
from pandera.typing.pandas import DataFrameModel, _GenericAlias

try:
    import pyspark.sql as ps

    PYSPARK_SQL_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_SQL_INSTALLED = False

if PYSPARK_SQL_INSTALLED:
    from pandera.engines import pyspark_engine

    PysparkString = pyspark_engine.String
    PysparkInt = pyspark_engine.Int
    PysparkLongInt = pyspark_engine.BigInt
    PysparkShortInt = pyspark_engine.ShortInt
    PysparkByteInt = pyspark_engine.ByteInt
    PysparkDouble = pyspark_engine.Double
    PysparkFloat = pyspark_engine.Float
    PysparkDecimal = pyspark_engine.Decimal
    PysparkDate = pyspark_engine.Date
    PysparkTimestamp = pyspark_engine.Timestamp
    PysparkBinary = pyspark_engine.Binary

    PysparkDType = TypeVar(  # type: ignore
        "PysparkDType",
        bound=Union[
            PysparkString,  # type: ignore
            PysparkInt,  # type: ignore
            PysparkLongInt,  # type: ignore
            PysparkShortInt,  # type: ignore
            PysparkByteInt,  # type: ignore
            PysparkDouble,  # type: ignore
            PysparkFloat,  # type: ignore
            PysparkDecimal,  # type: ignore
            PysparkDate,  # type: ignore
            PysparkTimestamp,  # type: ignore
            PysparkBinary,  # type: ignore
        ],
    )
    from typing import TYPE_CHECKING, Generic

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
                return _GenericAlias(cls, item)  # pragma: no cover
