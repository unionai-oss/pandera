"""Core pandera schema class definitions."""

import pandas as pd

from typing import Optional

from . import errors, constants
from .checks import Check


class DataFrameSchema(object):
    """A light-weight pandas DataFrame validator."""

    def __init__(
            self,
            columns,
            checks: callable = None,
            index=None,
            transformer: callable = None,
            coerce: bool = False,
            strict=False):
        """Initialize DataFrameSchema validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :type columns: dict[str -> Column]
        :param checks: dataframe-wide checks.
        :type checks: list[Check].
        :param index: specify the datatypes and properties of the index.
        :type index: Index
        :param transformer: a callable with signature:
            pandas.DataFrame -> pandas.DataFrame. If specified, calling
            `validate` will verify properties of the columns and return the
            transformed dataframe object.
        :type transformer: callable
        :param coerce: whether or not to coerce all of the columns on
            validation.
        :type coerce: bool
        :param strict: whether or not to accept columns in the dataframe that
            aren't in the DataFrameSchema.
        :type strict: bool
        """
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self._checks = checks
        self.index = index
        self.columns = columns
        self.transformer = transformer
        self.coerce = coerce
        self.strict = strict
        self._validate_schema()

    def __call__(
            self,
            dataframe: pd.DataFrame,
            head: int = None,
            tail: int = None,
            sample: int = None,
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

    def _validate_schema(self):

        for column_name, column in self.columns.items():
            for check in column._checks:
                if check.groupby is None or callable(check.groupby):
                    continue
                nonexistent_groupby_columns = [
                    c for c in check.groupby if c not in self.columns]
                if nonexistent_groupby_columns:
                    raise errors.SchemaInitError(
                        "groupby argument %s in Check for Column %s not "
                        "specified in the DataFrameSchema." %
                        (nonexistent_groupby_columns, column_name))

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
        return dataframe if len(dataframe_subsample) == 0 else \
            pd.concat(dataframe_subsample).drop_duplicates()

    def _check_dataframe(self, dataframe: pd.DataFrame):
        return all(
            check(self, check_index, check.prepare_dataframe_input(dataframe))
            for check_index, check in enumerate(self._checks))

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
        """
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
            col.set_name(col_name) for col_name, col in self.columns.items()
            if col.required or col_name in dataframe
        ]
        if self.index is not None:
            schema_components += [self.index]

        dataframe_to_validate = self._dataframe_to_validate(
            dataframe, head, tail, sample, random_state)
        assert (
            all(s(dataframe_to_validate) for s in schema_components) and
            self._check_dataframe(dataframe))
        if self.transformer is not None:
            dataframe = self.transformer(dataframe)
        return dataframe


class SeriesSchemaBase(object):
    """Base series validator object."""

    def __init__(
            self,
            pandas_dtype,
            checks: callable = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            name: str = None):
        """Initialize series schema base object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :type pandas_dtype: str|PandasDtype
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
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self._checks = checks
        self._name = name

        for check in self._checks:
            if check.groupby is not None and not self._allow_groupby:
                raise errors.SchemaInitError(
                    "Cannot use groupby checks with type %s" % type(self))

    @property
    def _allow_groupby(self):
        raise NotImplementedError(
            "The _allow_groupby property must be implemented by subclasses "
            "of SeriesSchemaBase")

    def __call__(
            self, series: pd.Series, dataframe: pd.DataFrame = None):
        """Validate a series."""
        if series.name != self._name:
            raise errors.SchemaError(
                "Expected %s to have name '%s', found '%s'" %
                (type(self), self._name, series.name))
        expected_dtype = _dtype = self._pandas_dtype if \
            isinstance(self._pandas_dtype, str) else self._pandas_dtype.value
        if self._nullable:
            series = series.dropna()
            if dataframe is not None:
                dataframe = dataframe.loc[series.index]
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
        else:
            nulls = series.isnull()
            if nulls.sum() > 0:
                if series.dtype != _dtype:
                    raise errors.SchemaError(
                        "expected series '%s' to have type %s, got %s and "
                        "non-nullable series contains null values: %s" %
                        (series.name, self._pandas_dtype.value, series.dtype,
                         series[nulls].head(
                            constants.N_FAILURE_CASES).to_dict()))
                else:
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

        if series.dtype != _dtype:
            raise errors.SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, expected_dtype, series.dtype))

        return all(
            check(
                self,
                check_index,
                check.prepare_series_input(series, dataframe))
            for check_index, check in enumerate(self._checks))


class SeriesSchema(SeriesSchemaBase):

    def __init__(
            self,
            pandas_dtype,
            checks: callable = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            name: str = None):
        """Initialize series schema object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :type pandas_dtype: str|PandasDtype
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
        super(SeriesSchema, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates, name)

    @property
    def _allow_groupby(self):
        return False

    def validate(self, series: pd.Series) -> pd.Series:
        """Check if all values in a series have a corresponding column in the
            DataFrameSchema

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).

        """
        if not isinstance(series, pd.Series):
            raise TypeError("expected %s, got %s" % (pd.Series, type(series)))
        if super(SeriesSchema, self).__call__(series):
            return series
        raise errors.SchemaError()
