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
            required: bool = True,
            name: str = None):
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
        self._name = name

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

    def validate(self, check_obj: pd.DataFrame) -> pd.DataFrame:
        """Validate a Column in a DataFrame object.

        :param check_obj: pandas DataFrame to validate.
        :returns: validated DataFrame.
        """
        if self._name is None:
            raise errors.SchemaError(
                "column name is set to None. Pass the ``name` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.")

        if self.coerce:
            check_obj[self.name] = self.coerce_dtype(
                check_obj[self.name])

        return super(Column, self).validate(check_obj)

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

    def coerce_dtype(self, series_or_index: pd.Index) -> pd.Index:
        """Coerce type of a pd.Index by type specified in pandas_dtype.

        :param pd.Index series: One-dimensional ndarray with axis labels
            (including time series).
        :returns: ``Index`` with coerced data type
        """
        if self._pandas_dtype is PandasDtype.String:
            return series_or_index.where(
                series_or_index.isna(), series_or_index.astype(str))
            # only coerce non-null elements to string
        return series_or_index.astype(self.dtype)

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def validate(
            self,
            check_obj: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Validate DataFrameSchema or SeriesSchema Index.

        :check_obj: pandas DataFrame of Series containing index to validate.
        :returns: validated DataFrame or Series.
        """

        if self.coerce:
            check_obj.index = self.coerce_dtype(check_obj.index)

        assert isinstance(
            super(Index, self).validate(pd.Series(check_obj.index)),
            pd.Series
        )
        return check_obj

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
        self.indexes = indexes
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

    @property
    def coerce(self):
        return self._coerce or any(index.coerce for index in self.indexes)

    def coerce_dtype(self, multi_index: pd.MultiIndex) -> pd.MultiIndex:
        """Coerce type of a pd.Series by type specified in pandas_dtype.

        :param multi_index: multi-index to coerce.
        :returns: ``MultiIndex`` with coerced data type
        """
        _coerced_multi_index = []
        if multi_index.nlevels != len(self.indexes):
            raise errors.SchemaError(
                "multi_index does not have equal number of levels as "
                "MultiIndex schema %d != %d." % (
                    multi_index.nlevels, len(self.indexes))
            )

        for level_i, index in enumerate(self.indexes):
            index_array = multi_index.get_level_values(level_i)
            if index.coerce or self.coerce:
                index_array = index.coerce_dtype(index_array)
            _coerced_multi_index.append(index_array)

        return pd.MultiIndex.from_arrays(
            _coerced_multi_index, names=multi_index.names)

    def validate(
            self,
            check_obj: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        # pylint: disable=signature-differs,arguments-differ
        # will need to clean up the class structure of this module since
        # this MultiIndex subclasses DataFrameSchema, which has a different
        # signature
        """Validate DataFrame or Series MultiIndex.

        :check_obj: pandas DataFrame of Series to validate.
        :returns: validated DataFrame or Series.
        """

        if self.coerce:
            check_obj.index = self.coerce_dtype(check_obj.index)

        assert isinstance(
            super(MultiIndex, self).validate(check_obj.index.to_frame()),
            pd.DataFrame
        )
        return check_obj

    def __repr__(self):
        return "<Schema MultiIndex: '%s'>" % list(self.columns)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
