"""Components used in pandera schemas."""

from typing import Union

import pandas as pd

from . import errors
from .dtypes import PandasDtype
from .checks import Check, List
from .schemas import DataFrameSchema, SeriesSchemaBase


class Column(SeriesSchemaBase):

    def __init__(
            self,
            pandas_dtype: Union[str, PandasDtype] = None,
            checks: Union[Check, List[Check]] = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            required: bool = True):
        """Create column validator object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :type pandas_dtype: str|PandasDtype
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates: Whether or not to coerce the column to the
            specified pandas_dtype before validation
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype.
        :param required: Whether or not column is allowed to be missing

        :raises SchemaInitError: if impossible to build schema from parameters

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>> from pandera import DataFrameSchema, Column
        >>>
        >>> schema = DataFrameSchema({
        ...     "column": Column(pa.String)
        ... })
        >>>
        >>> schema.validate(pd.DataFrame({"column": ["foo", "bar"]}))
          column
        0    foo
        1    bar

        See :ref:`here<column>` for more usage details.
        """
        super(Column, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates)
        self.coerce = coerce
        self.required = required
        self.pandas_dtype = pandas_dtype

        if coerce and pandas_dtype is None:
            raise errors.SchemaInitError(
                "Must specify dtype if coercing a Column's type")

    @property
    def _allow_groupby(self):
        return True

    def _set_name(self, name):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        self._name = name
        return self

    def _coerce_dtype(self, series):
        """Coerce the type of a pd.Series by the type specified in the Column
            object's self._pandas_dtype

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).

        """
        _dtype = str if self._pandas_dtype is PandasDtype.String \
            else self._pandas_dtype.value
        return series.astype(_dtype)

    def __call__(self, df):
        if self._name is None:
            raise RuntimeError(
                "need to `set_name` of column before calling it.")
        return super(Column, self).__call__(
            df[self._name], dataframe_context=df.drop(self._name, axis=1))

    def __repr__(self):
        if isinstance(self._pandas_dtype, PandasDtype):
            dtype = self._pandas_dtype.value
        else:
            dtype = self._pandas_dtype
        return "<Schema Column: '%s' type=%s>" % (self._name, dtype)


class Index(SeriesSchemaBase):

    def __init__(
            self,
            pandas_dtype: Union[str, PandasDtype] = None,
            checks: Union[Check, List[Check]] = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            name: str = None):
        """Create Index validator.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: checks to verify validity of the index.
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates: Whether or not to coerce the column to the
            specified pandas_dtype before validation
        :param name: name of the index

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>> from pandera import DataFrameSchema, Column, Index
        >>>
        >>> schema = DataFrameSchema(
        ...     columns={"column": Column(pa.String)},
        ...     index=Index(pa.Int, allow_duplicates=False))
        >>>
        >>> schema.validate(
        ...     pd.DataFrame({"column": ["foo"] * 3}, index=range(3))
        ... )
          column
        0    foo
        1    foo
        2    foo

        See :ref:`here<index>` for more usage details.

        """
        super(Index, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates, name)

    @property
    def _allow_groupby(self):
        return False

    def __call__(self, df):
        return super(Index, self).__call__(pd.Series(df.index))

    def __repr__(self):
        if self._name is None:
            return "<Schema Index>"
        return "<Schema Index: '%s'>" % self._name


class MultiIndex(DataFrameSchema):

    def __init__(
            self,
            indexes: List[Index],
            coerce: bool = False,
            strict=False):
        """Create MultiIndex validator.

        :param indexes: list of Index validators for each level of the
            MultiIndex index.
        :param coerce: Whether or not to coerce the MultiIndex to the
            specified pandas_dtypes before validation
        :param strict: whether or not to accept columns in the MultiIndex that
            aren't defined in the ``indexes`` argument.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> from pandera import Column, DataFrameSchema, MultiIndex
        >>>
        >>> schema = DataFrameSchema(
        ...     columns={"column": Column(pa.Int)},
        ...     index=MultiIndex([
        ...         Index(pa.String,
        ...               Check(lambda s: s.isin(["foo", "bar"])),
        ...               name="index0"),
        ...         Index(pa.Int, name="index1"),
        ...     ])
        ... )
        >>>
        >>> df = pd.DataFrame(
        ...     data={"column": [1, 2, 3]},
        ...     index=pd.MultiIndex(
        ...         levels=[["foo", "bar"], [0, 1, 2, 3, 4]],
        ...         labels=[[0, 1, 0], [0, 1, 2]],
        ...         names=["index0", "index1"],
        ...     )
        ... )
        >>>
        >>> schema.validate(df)
                       column
        index0 index1
        foo    0            1
        bar    1            2
        foo    2            3

        See :ref:`here<multiindex>` for more usage details.

        """
        super(MultiIndex, self).__init__(
            columns={
                i if index._name is None else index._name: Column(
                    pandas_dtype=index._pandas_dtype,
                    checks=index._checks,
                    nullable=index._nullable,
                    allow_duplicates=index._allow_duplicates,
                )
                for i, index in enumerate(indexes)
            },
            coerce=coerce,
            strict=strict,
        )

    def __call__(self, df):
        return isinstance(
            super(MultiIndex, self).__call__(df.index.to_frame()),
            pd.DataFrame
        )

    def __repr__(self):
        return "<Schema MultiIndex: '%s'>" % [c for c in self.columns]
