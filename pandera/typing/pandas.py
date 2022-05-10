"""Typing definitions and helpers."""
# pylint:disable=abstract-method,disable=too-many-ancestors
import io
from typing import _type_check, Type  # type: ignore[attr-defined]
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pandas as pd

from ..errors import SchemaError, SchemaInitError
from .common import DataFrameBase, GenericDtype, IndexBase, Schema, SeriesBase
from .formats import Formats

try:
    from typing import _GenericAlias  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _GenericAlias = None


try:
    from pydantic.fields import ModelField
except ImportError:
    ModelField = Any  # type: ignore


# pylint:disable=too-few-public-methods
class Index(IndexBase, pd.Index, Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """


# pylint:disable=too-few-public-methods
class Series(SeriesBase, pd.Series, Generic[GenericDtype]):  # type: ignore
    """Representation of pandas.Series, only used for type annotation.

    *new in 0.5.0*
    """

    if hasattr(pd.Series, "__class_getitem__") and _GenericAlias:

        def __class_getitem__(cls, item):
            """Define this to override the patch that pyspark.pandas performs on pandas.
            https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
            """
            _type_check(item, "Parameters to generic types must be types.")
            return _GenericAlias(cls, item)


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = Schema


# pylint:disable=too-few-public-methods
class DataFrame(DataFrameBase, pd.DataFrame, Generic[T]):
    """
    A generic type for pandas.DataFrame.

    *new in 0.5.0*
    """

    if hasattr(pd.DataFrame, "__class_getitem__") and _GenericAlias:

        def __class_getitem__(cls, item):
            """Define this to override the patch that pyspark.pandas performs on pandas.
            https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
            """
            _type_check(item, "Parameters to generic types must be types.")
            return _GenericAlias(cls, item)

    @classmethod
    def __get_validators__(cls):
        yield cls.pydantic_validate

    @classmethod
    def from_format(cls, obj: Any, config) -> pd.DataFrame:
        """
        Converts serialized data from a specific format
        specified in the :py:class:`pandera.model.SchemaModel` config options
        ``from_format`` and ``from_format_kwargs``.

        :param obj: object representing a serialized dataframe.
        :param config: schema model configuration object.
        """
        if config.from_format is None:
            if not isinstance(obj, pd.DataFrame):
                try:
                    obj = pd.DataFrame(obj)
                except Exception as exc:
                    raise ValueError(
                        f"Expected pd.DataFrame, found {type(obj)}"
                    ) from exc
            return obj

        reader = {
            Formats.dict: pd.DataFrame,
            Formats.csv: pd.read_csv,
            Formats.json: pd.read_json,
            Formats.feather: pd.read_feather,
            Formats.parquet: pd.read_parquet,
            Formats.pickle: pd.read_pickle,
        }[Formats(config.from_format)]

        return reader(obj, **(config.from_format_kwargs or {}))

    @classmethod
    def to_format(cls, data: pd.DataFrame, config) -> Any:
        """
        Converts a dataframe to the format specified in the
        :py:class:`pandera.model.SchemaModel` config options ``to_format``
        and ``to_format_kwargs``.

        :param data: convert this data to the specified format
        :param config: :py:cl
        """
        if config.to_format is None:
            return data

        writer, buffer = {
            Formats.dict: (data.to_dict, None),
            Formats.csv: (data.to_csv, None),
            Formats.json: (data.to_json, None),
            Formats.feather: (data.to_feather, io.BytesIO()),
            Formats.parquet: (data.to_parquet, io.BytesIO()),
            Formats.pickle: (data.to_pickle, io.BytesIO()),
        }[Formats(config.to_format)]

        args = [] if buffer is None else [buffer]
        out = writer(*args, **(config.to_format_kwargs or {}))
        if buffer is None:
            return out
        elif buffer.closed:
            raise IOError(
                f"pandas=={pd.__version__} closed the buffer automatically "
                f"using the serialization method {writer}. Use a later "
                "version of pandas or use a different the serialization "
                "format."
            )
        buffer.seek(0)
        return buffer

    @classmethod
    def _get_schema(cls, field: ModelField):
        if not field.sub_fields:
            raise TypeError(
                "Expected a typed pandera.typing.DataFrame,"
                " e.g. DataFrame[Schema]"
            )
        schema_model = field.sub_fields[0].type_
        try:
            schema = schema_model.to_schema()
        except SchemaInitError as exc:
            raise ValueError(
                f"Cannot use {cls.__name__} as a pydantic type as its "
                "SchemaModel cannot be converted to a DataFrameSchema.\n"
                f"Please revisit the model to address the following errors:"
                f"\n{exc}"
            ) from exc
        return schema_model, schema

    @classmethod
    def pydantic_validate(cls, obj: Any, field: ModelField) -> pd.DataFrame:
        """
        Verify that the input can be converted into a pandas dataframe that
        meets all schema requirements.
        """
        schema_model, schema = cls._get_schema(field)
        data = cls.from_format(obj, schema_model.__config__)

        try:
            valid_data = schema.validate(data)
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc

        return cls.to_format(valid_data, schema_model.__config__)

    @staticmethod
    def from_records(
            type: Type[T],
            data,
            index=None,
            exclude=None,
            columns=None,
            coerce_float: bool = False,
            nrows: int | None = None,
    ) -> "DataFrame[T]":
        """
        Convert structured or record ndarray to DataFrame.

        Creates a DataFrame object from a structured ndarray, sequence of
        tuples or dicts, or DataFrame.

        Parameters
        ----------
        type: the type of the schema model for the datarecords, the index columns are resolved from the
            schema when check_name=True
        data : structured ndarray, sequence of tuples or dicts, or DataFrame
            Structured input data.
        index : str, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use.
        exclude : sequence, default None
            Columns or fields to exclude.
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns).
        coerce_float : bool, default False
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        nrows : int, default None
            Number of rows to read if data is an iterator.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_dict : DataFrame from dict of array-like or dicts.
        DataFrame : DataFrame object creation using constructor.

        Examples
        --------
        Data can be provided as a structured ndarray:

        >>> data = np.array([(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')],
        ...                 dtype=[('col_1', 'i4'), ('col_2', 'U1')])
        >>> pd.DataFrame.from_records(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Data can be provided as a list of dicts:

        >>> data = [{'col_1': 3, 'col_2': 'a'},
        ...         {'col_1': 2, 'col_2': 'b'},
        ...         {'col_1': 1, 'col_2': 'c'},
        ...         {'col_1': 0, 'col_2': 'd'}]
        >>> pd.DataFrame.from_records(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Data can be provided as a list of tuples with corresponding columns:

        >>> data = [(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')]
        >>> pd.DataFrame.from_records(data, columns=['col_1', 'col_2'])
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d
        """
        schema = type.to_schema
        if index is None:
            index = schema.index.names

        return schema.validate(pd.DataFrame.from_records(data, index, exclude, columns, coerce_float, nrows))

