"""Core pandera schema class definitions."""

import json
import copy
import warnings
from typing import List, Optional, Union, Dict, Any

import pandas as pd

from . import errors, constants, dtypes, error_formatters
from .checks import Check


N_INDENT_SPACES = 4


class DataFrameSchema():
    """A light-weight pandas DataFrame validator."""

    def __init__(
            self,
            columns: Dict[str, Any] = None,
            checks: Optional[List[Check]] = None,
            index=None,
            transformer: callable = None,
            coerce: bool = False,
            strict=False):
        """Initialize DataFrameSchema validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :type columns: mapping of column names and column schema component.
        :param checks: dataframe-wide checks.
        :param index: specify the datatypes and properties of the index.
        :param transformer: a callable with signature:
            pandas.DataFrame -> pandas.DataFrame. If specified, calling
            `validate` will verify properties of the columns and return the
            transformed dataframe object.
        :param coerce: whether or not to coerce all of the columns on
            validation.
        :param strict: whether or not to accept columns in the dataframe that
            aren't in the DataFrameSchema.

        :raises SchemaInitError: if impossible to build schema from parameters

        :examples:

        >>> import pandera as pa
        >>> from pandera import DataFrameSchema, Column
        >>>
        >>> schema = DataFrameSchema({
        ...     "str_column": Column(pa.String),
        ...     "float_column": Column(pa.Float),
        ...     "int_column": Column(pa.Int),
        ...     "date_column": Column(pa.DateTime),
        ... })

        Use the pandas API to define checks, which takes a function with
        the signature: ``pd.Series -> Union[bool, pd.Series]`` where the
        output series contains boolean values.

        >>> from pandera import Check
        >>>
        >>> schema_withchecks = DataFrameSchema({
        ...     "probability": Column(
        ...         pa.Float, Check(lambda s: (s >= 0) & (s <= 1))),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": Column(
        ...         pa.String, [
        ...             Check(lambda s: s.isin(["dog", "cat", "duck"])),
        ...             Check(lambda s: (s == "dog").mean() > 0.5),
        ...         ]),
        ... })

        See :ref:`here<DataFrameSchemas>` for more usage details.

        """
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]

        self.columns = {} if columns is None else columns

        if coerce and None in [c.pandas_dtype for c in self.columns.values()]:
            raise errors.SchemaInitError(
                "Must specify dtype in all Columns if coercing "
                "DataFrameSchema")

        self.checks = checks
        self.index = index
        self.transformer = transformer
        self.strict = strict
        self._coerce = coerce
        self._validate_schema()
        self._set_column_names()

    @property
    def coerce(self):
        """Whether to coerce series to specified type."""
        return self._coerce

    def _validate_schema(self):
        for column_name, column in self.columns.items():
            for check in column.checks:
                if check.groupby is None or callable(check.groupby):
                    continue
                nonexistent_groupby_columns = [
                    c for c in check.groupby if c not in self.columns]
                if nonexistent_groupby_columns:
                    raise errors.SchemaInitError(
                        "groupby argument %s in Check for Column %s not "
                        "specified in the DataFrameSchema." %
                        (nonexistent_groupby_columns, column_name))

    def _set_column_names(self):

        def _set_column_handler(column, column_name):
            if column.name is not None and column.name != column_name:
                warnings.warn(
                    "resetting column for %s to '%s'." % (column, column_name))
            elif column.name == column_name:
                return column
            return column.set_name(column_name)

        self.columns = {
            column_name: _set_column_handler(column, column_name)
            for column_name, column in self.columns.items()
        }

    @staticmethod
    def _dataframe_to_validate(
            dataframe: pd.DataFrame,
            head: int,
            tail: int,
            sample: int,
            random_state: Optional[int]) -> pd.DataFrame:
        dataframe_subsample = []
        if head is not None:
            dataframe_subsample.append(dataframe.head(head))
        if tail is not None:
            dataframe_subsample.append(dataframe.tail(tail))
        if sample is not None:
            dataframe_subsample.append(
                dataframe.sample(sample, random_state=random_state))
        return dataframe if not dataframe_subsample else \
            pd.concat(dataframe_subsample).drop_duplicates()

    def _check_dataframe(self, dataframe):
        check_results = []
        for check_index, check in enumerate(self.checks):
            check_results.append(
                _handle_check_results(self, check_index, check, dataframe)
            )
        return all(check_results)

    @property
    def dtype(self) -> Dict[str, str]:
        """A pandas style dtype dict where the keys are column names and values
        are pandas dtype for the column
        """
        return {k: v.dtype for k, v in self.columns.items()}

    def validate(
            self,
            dataframe: pd.DataFrame,
            head: int = None,
            tail: int = None,
            sample: int = None,
            random_state: Optional[int] = None) -> pd.DataFrame:
        """Check if all columns in a dataframe have a column in the Schema.

        :param pd.DataFrame dataframe: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :type head: int
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :type tail: int
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :returns: validated ``DataFrame``

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        Calling ``schema.validate`` returns the dataframe.

        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     "probability": [0.1, 0.4, 0.52, 0.23, 0.8, 0.76],
        ...     "category": ["dog", "dog", "cat", "duck", "dog", "dog"]
        ... })
        >>>
        >>> schema_withchecks.validate(df)[["probability", "category"]]
           probability category
        0         0.10      dog
        1         0.40      dog
        2         0.52      cat
        3         0.23     duck
        4         0.80      dog
        5         0.76      dog
        """
        dataframe = dataframe.copy()

        if self.strict:
            for column in dataframe:
                if column not in self.columns:
                    raise errors.SchemaError(
                        "column '%s' not in DataFrameSchema %s" %
                        (column, self.columns)
                    )

        for colname, col in self.columns.items():
            if colname not in dataframe and col.required:
                raise errors.SchemaError(
                    "column '%s' not in dataframe\n%s" %
                    (colname, dataframe.head()))

            if col.coerce or self.coerce:
                dataframe[colname] = col.coerce_dtype(dataframe[colname])

        schema_components = [
            col for col_name, col in self.columns.items()
            if col.required or col_name in dataframe
        ]
        if self.index is not None:
            if self.index.coerce or self.coerce:
                dataframe.index = self.index.coerce_dtype(dataframe.index)
            schema_components.append(self.index)

        dataframe_to_validate = self._dataframe_to_validate(
            dataframe, head, tail, sample, random_state)

        assert (
            all(isinstance(component(dataframe_to_validate), pd.DataFrame)
                for component in schema_components)
            and self._check_dataframe(dataframe_to_validate))

        if self.transformer is not None:
            dataframe = self.transformer(dataframe)

        return dataframe

    def __call__(
            self,
            dataframe: pd.DataFrame,
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None):
        """Delegate to `validate` method.

        :param pd.DataFrame dataframe: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :type head: int
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :type tail: int
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        """
        return self.validate(dataframe)

    def __repr__(self):
        """Represent string for logging."""
        return "%s(columns=%s, index=%s, transformer=%s, coerce=%s)" % \
            (self.__class__.__name__,
             self.columns,
             self.index,
             self.transformer,
             self.coerce)

    def __str__(self):
        """Represent string for user inspection."""
        columns = {k: str(v) for k, v in self.columns.items()}
        columns = json.dumps(columns, indent=N_INDENT_SPACES)
        _indent = " " * N_INDENT_SPACES
        columns = "\n".join(
            "{}{}".format(_indent, line) if i != 0
            else "{}columns={}".format(_indent, line)
            for i, line in enumerate(columns.split("\n")))
        return (
            "{class_name}(\n"
            "{columns},\n"
            "{indent}index={index},\n"
            "{indent}transformer={transformer},\n"
            "{indent}coerce={coerce},\n"
            "{indent}strict={strict}\n"
            ")"
        ).format(
            class_name=self.__class__.__name__,
            columns=columns,
            index=str(self.index),
            transformer=str(self.transformer),
            coerce=self.coerce,
            strict=self.strict,
            indent=_indent,
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def add_columns(self,
                    extra_schema_cols: Dict[str, Any]) -> 'DataFrameSchema':
        """Create a new DataFrameSchema with extra Columns

        :param extra_schema_cols: Additional columns of the format
        :type extra_schema_cols: DataFrameSchema
        :returns: a new DataFrameSchema with the extra_schema_cols added

        """
        schema_copy = copy.deepcopy(self)
        schema_copy.columns = {**schema_copy.columns,
                               **DataFrameSchema(extra_schema_cols).columns}
        return schema_copy

    def remove_columns(self,
                       cols_to_remove: List) -> 'DataFrameSchema':
        """Removes a column from a DataFrameSchema and returns a new
        DataFrameSchema.

        :param cols_to_remove: Columns to be removed from the DataFrameSchema
        :type cols_to_remove: List
        :returns: a new DataFrameSchema without the cols_to_remove

        """
        schema_copy = copy.deepcopy(self)
        for col in cols_to_remove:
            schema_copy.columns.pop(col)

        return schema_copy


class SeriesSchemaBase():
    """Base series validator object."""

    def __init__(
            self,
            pandas_dtype: Union[str, dtypes.PandasDtype] = None,
            checks: callable = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            name: str = None):
        """Initialize series schema base object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        :type checks: callable
        :param nullable: Whether or not column can contain null values.
        :type nullable: bool
        :param allow_duplicates:
        :type allow_duplicates: bool
        """
        self._pandas_dtype = pandas_dtype
        self._nullable = nullable
        self._allow_duplicates = allow_duplicates
        self._coerce = coerce
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self.checks = checks
        self._name = name

        for check in self.checks:
            if check.groupby is not None and not self._allow_groupby:
                raise errors.SchemaInitError(
                    "Cannot use groupby checks with type %s" % type(self))

    @property
    def coerce(self) -> bool:
        """Whether to coerce series to specified type."""
        return self._coerce

    @property
    def name(self) -> str:
        """Get SeriesSchema name."""
        return self._name

    @property
    def dtype(self) -> str:
        """String representation of the dtype."""
        if isinstance(self._pandas_dtype, str) or self._pandas_dtype is None:
            dtype = self._pandas_dtype
        elif self._pandas_dtype is dtypes.PandasDtype.String:
            # handle special case of string.
            dtype = dtypes.PandasDtype.Object.value
        else:
            dtype = self._pandas_dtype.value
        return dtype

    def coerce_dtype(
            self, series_or_index: Union[pd.Series, pd.Index]) -> pd.Series:
        """Coerce type of a pd.Series by type specified in pandas_dtype.

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).
        :returns: ``Series`` with coerced data type
        """
        if self._pandas_dtype is dtypes.PandasDtype.String:
            # only coerce non-null elements to string
            return series_or_index.where(
                series_or_index.isna(), series_or_index.astype(str))
        return series_or_index.astype(self.dtype)

    @property
    def _allow_groupby(self):
        """Whether the schema or schema component allows groupby operations."""
        raise NotImplementedError(
            "The _allow_groupby property must be implemented by subclasses "
            "of SeriesSchemaBase")

    def validate(
            self,
            check_obj: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        # pylint: disable=too-many-branches,W0212
        """Validate a series or specific column in dataframe.

        :check_obj: pandas DataFrame or Series to validate.
        :returns: validated DataFrame or Series.

        """
        series = check_obj.copy() if isinstance(check_obj, pd.Series) \
            else check_obj[self.name].copy()

        if series.name != self._name:
            raise errors.SchemaError(
                "Expected %s to have name '%s', found '%s'" %
                (type(self), self._name, series.name))

        _dtype = self.dtype

        if self._nullable:
            series = series.dropna()
            if _dtype in ["int_", "int8", "int16", "int32", "int64", "uint8",
                          "uint16", "uint32", "uint64"]:
                _series = series.astype(_dtype)
                if (_series != series).any():
                    # in case where dtype is meant to be int, make sure that
                    # casting to int results in the same values.
                    raise errors.SchemaError(
                        "after dropping null values, expected values in "
                        "series '%s' to be int, found: %s" %
                        (series.name, set(series)))
                series = _series

        nulls = series.isnull()
        if nulls.sum() > 0:
            if series.dtype != _dtype:
                raise errors.SchemaError(
                    "expected series '%s' to have type %s, got %s and "
                    "non-nullable series contains null values: %s" %
                    (series.name, self._pandas_dtype.value, series.dtype,
                     series[nulls].head(
                         constants.N_FAILURE_CASES).to_dict()))
            raise errors.SchemaError(
                "non-nullable series '%s' contains null values: %s" %
                (series.name,
                 series[nulls].head(
                     constants.N_FAILURE_CASES).to_dict()))

        # Check if the series contains duplicate values
        if not self._allow_duplicates:
            duplicates = series.duplicated()
            if any(duplicates):
                raise errors.SchemaError(
                    "series '%s' contains duplicate values: %s" %
                    (series.name,
                     series[duplicates].head(
                         constants.N_FAILURE_CASES).to_dict()))

        try:
            series.dtype == _dtype
        except TypeError:
            types_not_matching = True
        else:
            types_not_matching = series.dtype != _dtype

        if _dtype is not None and types_not_matching:
            raise errors.SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, _dtype, series.dtype))

        check_results = []

        if isinstance(check_obj, pd.Series):
            check_args = (series, )
        else:
            _check_obj = check_obj.loc[series.index].copy()
            _check_obj[self.name] = series
            check_args = (_check_obj, self.name)

        for check_index, check in enumerate(self.checks):
            check_results.append(
                _handle_check_results(self, check_index, check, *check_args)
            )

        assert all(check_results)
        return check_obj

    def __call__(
            self,
            check_obj: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Validate a series or column in a dataframe."""
        return self.validate(check_obj)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SeriesSchema(SeriesSchemaBase):
    """Series validator."""

    def __init__(
            self,
            pandas_dtype: dtypes.PandasDtype = None,
            checks: List[Check] = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            name: str = None):
        """Initialize series schema object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates:
        :param coerce: whether or not to coerce all of the columns on
            validation.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> from pandera import SeriesSchema
        >>>
        >>>
        >>> series_schema = SeriesSchema(
        ...     pa.Float, [
        ...         Check(lambda s: s > 0),
        ...         Check(lambda s: s < 1000),
        ...         Check(lambda s: s.mean() > 300),
        ...     ])

        See :ref:`here<SeriesSchemas>` for more usage details.
        """
        super(SeriesSchema, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates, coerce, name)

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def validate(self, check_obj: pd.Series) -> pd.Series:
        """Validate a Series object.

        :param check_obj: One-dimensional ndarray with axis labels
            (including time series).
        :returns: validated Series.

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        >>> series = pd.Series([1, 100, 800, 900, 999], dtype=float)
        >>> print(series_schema.validate(series))
        0      1.0
        1    100.0
        2    800.0
        3    900.0
        4    999.0
        dtype: float64

        """
        if not isinstance(check_obj, pd.Series):
            raise TypeError(
                "expected %s, got %s" % (pd.Series, type(check_obj)))

        if self.coerce:
            check_obj = self.coerce_dtype(check_obj)

        return super(SeriesSchema, self).validate(check_obj)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _handle_check_results(
        schema: Union[DataFrameSchema, SeriesSchemaBase],
        check_index: int,
        check: Check,
        *check_args) -> bool:
    """Handle check results, raising SchemaError on check failure.

    :param check_index: index of check in the schema component check list.
    :param check: Check object used to validate pandas object.
    :param check_args: arguments to pass into check object.
    """
    check_result = check(*check_args)
    if not check_result.check_passed:
        if check_result.failure_cases is None:
            raise errors.SchemaError(
                error_formatters.format_generic_error_message(
                    schema, check, check_index))
        raise errors.SchemaError(
            error_formatters.format_vectorized_error_message(
                schema, check, check_index, check_result.failure_cases)
        )
    return check_result.check_passed
