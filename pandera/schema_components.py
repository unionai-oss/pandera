"""Components used in pandera schemas."""

from typing import Union

import pandas as pd

from . import errors
from .dtypes import PandasDtype
from .checks import Check, List
from .schemas import DataFrameSchema, SeriesSchemaBase


class Column(SeriesSchemaBase):
    """Extends SeriesSchemaBase with Column-specific options"""

    def __init__(
            self,
            pandas_dtype: Union[str, PandasDtype] = None,
            checks: Union[Check, List[Check]] = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            required: bool = True):
        """Create column validator object.

        :param pandas_dtype: datatype of the column. A ``PandasDtype`` for
            type-checking dataframe. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
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
            pandas_dtype, checks, nullable, allow_duplicates, coerce)
        self.required = required
        self.pandas_dtype = pandas_dtype

        if coerce and pandas_dtype is None:
            raise errors.SchemaInitError(
                "Must specify dtype if coercing a Column's type")

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return True

    def set_name(self, name: str):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        self._name = name
        return self

    def __call__(self, df_or_series: Union[pd.DataFrame, pd.Series]) -> bool:
        """Validate DataFrameSchema Column."""
        if self._name is None:
            raise RuntimeError(
                "need to `set_name` of column before calling it.")
        return super(Column, self).__call__(df_or_series)

    def __repr__(self):
        if isinstance(self._pandas_dtype, PandasDtype):
            dtype = self._pandas_dtype.value
        else:
            dtype = self._pandas_dtype
        return "<Schema Column: '%s' type=%s>" % (self._name, dtype)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Index(SeriesSchemaBase):
    """Extends SeriesSchemaBase with Index-specific options"""

    def __init__(
            self,
            pandas_dtype: Union[str, PandasDtype] = None,
            checks: Union[Check, List[Check]] = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            name: str = None):
        """Create Index validator.

        :param pandas_dtype: datatype of the column. A ``PandasDtype`` for
            type-checking dataframe. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: checks to verify validity of the index.
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates: Whether or not to coerce the column to the
            specified pandas_dtype before validation
        :param coerce: If True, when schema.validate is called the index will
            be coerced into the specified dtype.
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
            pandas_dtype, checks, nullable, allow_duplicates, coerce, name)

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def __call__(self, df_or_series: Union[pd.DataFrame, pd.Series]) -> bool:
        """Validate DataFrameSchema Index."""
        return super(Index, self).__call__(pd.Series(df_or_series.index))

    def __repr__(self):
        if self._name is None:
            return "<Schema Index>"
        return "<Schema Index: '%s'>" % self._name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MultiIndex(DataFrameSchema):
    """Extends DataFrameSchema with Multi-index-specific options.

    Because `MultiIndex.__call__` converts the index to a dataframe via
    `to_frame()`, each index is treated as a series and it makes sense to
    inherit the `__call__` and `validate` methods from DataFrameSchema.
    """

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
        # pylint: disable=W0212
        super(MultiIndex, self).__init__(
            columns={
                i if index._name is None else index._name: Column(
                    pandas_dtype=index._pandas_dtype,
                    checks=index.checks,
                    nullable=index._nullable,
                    allow_duplicates=index._allow_duplicates,
                )
                for i, index in enumerate(indexes)
            },
            coerce=coerce,
            strict=strict,
        )

    def __call__(self, df_or_series: Union[pd.DataFrame, pd.Series]) -> bool:
        # pylint: disable=signature-differs,W0222
        # false positive warning is raised here, even though method signature
        # is exactly the same. Will need to investigate why this is being
        # raised.
        """Validate DataFrameSchema MultiIndex."""
        return isinstance(
            super(MultiIndex, self).__call__(df_or_series.index.to_frame()),
            pd.DataFrame
        )

    def __repr__(self):
        return "<Schema MultiIndex: '%s'>" % list(self.columns)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
