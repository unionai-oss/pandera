"""Components used in pandera schemas."""

import pandas as pd

from .dtypes import PandasDtype
from .schemas import DataFrameSchema, SeriesSchemaBase


class Column(SeriesSchemaBase):

    def __init__(
            self,
            pandas_dtype,
            checks: callable = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            required: bool = True):
        """Initialize column validator object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :type pandas_dtype: str|PandasDtype
        :param checks: if element_wise is True, then callable signature should
            be: x -> bool where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        :type checks: callable
        :param nullable: Whether or not column can contain null values.
        :type nullable: bool
        :param allow_duplicates: Whether or not to coerce the column to the
            specified pandas_dtype before validation
        :type allow_duplicates: bool
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype.
        :type coerce:  bool
        :param required: Whether or not column is allowed to be missing
        :type required:  bool
        """
        super(Column, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates)
        self.coerce = coerce
        self.required = required

    @property
    def _allow_groupby(self):
        return True

    def set_name(self, name):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        self._name = name
        return self

    def coerce_dtype(self, series):
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
            pandas_dtype,
            checks: callable = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            name: str = None):
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

    def __init__(self, indexes, coerce: bool = False, strict=False):
        super(MultiIndex, self).__init__(
            columns={
                i if index._name is None else index._name: Column(
                    index._pandas_dtype,
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
