"""Components used in pandera schemas."""

from copy import copy

from typing import Union, Optional, Tuple, Any, List, Dict

import numpy as np
import pandas as pd

from . import errors
from .dtypes import PandasDtype
from .schemas import (
    DataFrameSchema, SeriesSchemaBase, CheckList, PandasDtypeInputTypes
)


def _is_valid_multiindex_tuple_str(x: Tuple[Any]) -> bool:
    """Check that a multi-index tuple key has all string elements"""
    return isinstance(x, tuple) and all(isinstance(i, str) for i in x)


class Column(SeriesSchemaBase):
    """Validate types and properties of DataFrame columns."""

    def __init__(
            self,
            pandas_dtype: PandasDtypeInputTypes = None,
            checks: CheckList = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            required: bool = True,
            name: str = None,
            regex: bool = False,
    ) -> None:
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
        :param name: column name in dataframe to validate.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a dataframe.
        :raises SchemaInitError: if impossible to build schema from parameters

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "column": pa.Column(pa.String)
        ... })
        >>>
        >>> schema.validate(pd.DataFrame({"column": ["foo", "bar"]}))
          column
        0    foo
        1    bar

        See :ref:`here<column>` for more usage details.
        """
        super().__init__(
            pandas_dtype, checks, nullable, allow_duplicates, coerce
        )
        if name is not None \
                and not isinstance(name, str) \
                and not _is_valid_multiindex_tuple_str(name) \
                and regex:
            raise ValueError(
                "You cannot specify a non-string name when setting regex=True")
        self.required = required
        self._name = name
        self._regex = regex

        if coerce and self._pandas_dtype is None:
            raise errors.SchemaInitError(
                "Must specify dtype if coercing a Column's type")

    @property
    def regex(self) -> bool:
        """True if ``name`` attribute should be treated as a regex pattern."""
        return self._regex

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return True

    @property
    def properties(self) -> Dict[str, Any]:
        """Get column properties."""
        return {
            "pandas_dtype": self._pandas_dtype,
            "checks": self._checks,
            "nullable": self._nullable,
            "allow_duplicates": self._allow_duplicates,
            "coerce": self._coerce,
            "required": self.required,
            "name": self._name,
            "regex": self._regex,
        }

    def set_name(self, name: str):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        if not isinstance(name, str) and \
                not _is_valid_multiindex_tuple_str(name) \
                and self.regex:
            raise ValueError(
                "You cannot specify a non-string name when setting regex=True")
        self._name = name
        return self

    def validate(
            self,
            check_obj: pd.DataFrame,
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> pd.DataFrame:
        # pylint: disable=duplicate-code
        """Validate a Column in a DataFrame object.

        :param check_obj: pandas DataFrame to validate.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrorReport``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :returns: validated DataFrame.
        """
        check_obj = check_obj.copy()

        if self._name is None:
            raise errors.SchemaError(
                self, check_obj,
                "column name is set to None. Pass the ``name` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.")

        column_keys_to_check = self.get_regex_columns(check_obj.columns) if \
            self._regex else [self._name]

        check_results = []
        for column_name in column_keys_to_check:
            if self.coerce:
                check_obj[column_name] = self.coerce_dtype(
                    check_obj[column_name])
            check_results.append(
                isinstance(
                    super(Column, copy(self).set_name(column_name))
                    .validate(
                        check_obj, head, tail, sample, random_state, lazy
                    ),
                    pd.DataFrame)
            )

        assert all(check_results)
        return check_obj

    def get_regex_columns(
            self,
            columns: Union[pd.Index, pd.MultiIndex]
    ) -> Union[pd.Index, pd.MultiIndex]:
        """Get matching column names based on regex column name pattern.

        :param columns: columns to regex pattern match
        :returns: matchin columns
        """
        if isinstance(self.name, tuple):
            # handle MultiIndex case
            if len(self.name) != columns.nlevels:
                raise IndexError(
                    "Column regex name='%s' is a tuple, expected a MultiIndex "
                    "columns with %d number of levels, found %d level(s)" %
                    (self.name, len(self.name), columns.nlevels)
                )
            matches = np.ones(len(columns)).astype(bool)
            for i, name in enumerate(self.name):
                matched = pd.Index(
                    columns.get_level_values(i).str.match(name)
                ).fillna(False)
                matches = matches & np.array(matched.tolist())
            column_keys_to_check = columns[matches]
        else:
            if isinstance(columns, pd.MultiIndex):
                raise IndexError(
                    "Column regex name %s is a string, expected a dataframe "
                    "where the index is a pd.Index object, not a "
                    "pd.MultiIndex object" % (self.name)
                )
            column_keys_to_check = columns[
                # str.match will return nan values when the index value is
                # not a string.
                pd.Index(columns.str.match(self.name)).fillna(False).tolist()
            ]
        if column_keys_to_check.shape[0] == 0:
            raise errors.SchemaError(
                self, columns,
                "Column regex name='%s' did not match any columns in the "
                "dataframe. Update the regex pattern so that it matches at "
                "least one column:\n%s" % (self.name, columns.tolist())
            )
        return column_keys_to_check

    def __repr__(self):
        if isinstance(self._pandas_dtype, PandasDtype):
            dtype = self._pandas_dtype.value
        else:
            dtype = self._pandas_dtype
        return "<Schema Column: '%s' type=%s>" % (self._name, dtype)

    def __eq__(self, other):
        def _compare_dict(obj):
            return {
                k: v if k != "_checks" else set(v)
                for k, v in obj.__dict__.items()
            }
        return _compare_dict(self) == _compare_dict(other)


class Index(SeriesSchemaBase):
    """Validate types and properties of a DataFrame Index."""

    def __init__(
            self,
            pandas_dtype: PandasDtypeInputTypes = None,
            checks: CheckList = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            name: str = None) -> None:
        # pylint: disable=useless-super-delegation
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
        >>>
        >>>
        >>> schema = pa.DataFrameSchema(
        ...     columns={"column": pa.Column(pa.String)},
        ...     index=pa.Index(pa.Int, allow_duplicates=False))
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
        super().__init__(
            pandas_dtype, checks, nullable, allow_duplicates, coerce, name
        )

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
            check_obj: Union[pd.DataFrame, pd.Series],
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        # pylint: disable=duplicate-code
        """Validate DataFrameSchema or SeriesSchema Index.

        :check_obj: pandas DataFrame of Series containing index to validate.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :returns: validated DataFrame or Series.
        """

        if self.coerce:
            check_obj.index = self.coerce_dtype(check_obj.index)

        assert isinstance(
            super().validate(
                pd.Series(check_obj.index),
                head, tail, sample, random_state, lazy,
            ),
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
    """Validate types and properties of a DataFrame MultiIndex.

    Because `MultiIndex.__call__` converts the index to a dataframe via
    `to_frame()`, each index is treated as a series and it makes sense to
    inherit the `__call__` and `validate` methods from DataFrameSchema.
    """

    def __init__(
            self,
            indexes: List[Index],
            coerce: bool = False,
            strict: bool = False,
            name: str = None) -> None:
        """Create MultiIndex validator.

        :param indexes: list of Index validators for each level of the
            MultiIndex index.
        :param coerce: Whether or not to coerce the MultiIndex to the
            specified pandas_dtypes before validation
        :param strict: whether or not to accept columns in the MultiIndex that
            aren't defined in the ``indexes`` argument.
        :param name: name of schema component

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>>
        >>> schema = pa.DataFrameSchema(
        ...     columns={"column": pa.Column(pa.Int)},
        ...     index=pa.MultiIndex([
        ...         pa.Index(pa.String,
        ...               pa.Check(lambda s: s.isin(["foo", "bar"])),
        ...               name="index0"),
        ...         pa.Index(pa.Int, name="index1"),
        ...     ])
        ... )
        >>>
        >>> df = pd.DataFrame(
        ...     data={"column": [1, 2, 3]},
        ...     index=pd.MultiIndex.from_arrays(
        ...         [["foo", "bar", "foo"], [0, 1, 2]],
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
        super().__init__(
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
            name=name,
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
                self, multi_index,
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
            check_obj: Union[pd.DataFrame, pd.Series],
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        # pylint: disable=signature-differs,arguments-differ,duplicate-code
        # will need to clean up the class structure of this module since
        # this MultiIndex subclasses DataFrameSchema, which has a different
        # signature
        """Validate DataFrame or Series MultiIndex.

        :param check_obj: pandas DataFrame of Series to validate.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrorReport``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :returns: validated DataFrame or Series.
        """

        if self.coerce:
            check_obj.index = self.coerce_dtype(check_obj.index)

        try:
            validation_result = super().validate(
                check_obj.index.to_frame(),
                head, tail, sample, random_state, lazy,
            )
        except errors.SchemaErrors as err:
            # This is a hack to re-raise the SchemaErrors exception and change
            # the schema context to MultiIndex. This should be fixed by with
            # a more principled schema class hierarchy.
            schema_error_dicts = []
            # pylint: disable=protected-access
            for schema_error_dict in err._schema_error_dicts:
                error = schema_error_dict["error"]
                error = errors.SchemaError(
                    self, check_obj, error.args[0],
                    error.failure_cases.assign(column=error.schema.name),
                    error.check, error.check_index
                )
                schema_error_dict["error"] = error
                schema_error_dicts.append(schema_error_dict)

            raise errors.SchemaErrors(schema_error_dicts, check_obj)

        assert isinstance(validation_result, pd.DataFrame)
        return check_obj

    def __repr__(self):
        return "<Schema MultiIndex: '%s'>" % list(self.columns)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
