"""Validate Pandas Data Structures."""

import inspect
import sys
import warnings
import pandas as pd
import wrapt

from collections import OrderedDict
from enum import Enum


class SchemaError(Exception):
    pass


class PandasDtype(Enum):
    Bool = "bool"
    DateTime = "datetime64[ns]"
    Category = "category"
    Float = "float64"
    Int = "int64"
    Object = "object"
    String = "object"
    Timedelta = "timedelta64[ns]"


Bool = PandasDtype.Bool
DateTime = PandasDtype.DateTime
Category = PandasDtype.Category
Float = PandasDtype.Float
Int = PandasDtype.Int
Object = PandasDtype.Object
String = PandasDtype.String
Timedelta = PandasDtype.Timedelta

N_FAILURE_CASES = 10


class Check(object):

    def __init__(self, fn, element_wise=True, error=None, n_failure_cases=10):
        """Check object applies function element-wise or series-wise

        Parameters
        ----------
        fn : callable
            A function to check series schema. If element_wise is True,
            then callable signature should be: x -> bool where x is a
            scalar element in the column. Otherwise, signature is expected
            to be: pd.Series -> bool|pd.Series[bool].
        element_wise : bool|list[bool]
            Whether or not to apply validator in an element-wise fashion. If
            bool, assumes that all checks should be applied to the column
            element-wise. If list, should be the same number of elements
            as checks.
        error : str
            custom error message if series fails validation check.
        n_failure_cases : int|None
            report the top n failure cases. If None, then report all failure
            cases.
        """
        self.fn = fn
        self.element_wise = element_wise
        self.error = error
        self.n_failure_cases = n_failure_cases

    @property
    def error_message(self):
        if self.error:
            return "%s: %s" % (self.fn.__name__, self.error)
        return "%s" % self.fn.__name__

    def vectorized_error_message(self, parent_schema, index, failure_cases):
        return (
                "%s failed element-wise validator %d:\n"
                "%s\nfailure cases:\n%s" %
                (parent_schema, index,
                 self.error_message,
                 self._format_failure_cases(failure_cases)))

    def generic_error_message(self, parent_schema, index):
        return "%s failed series validator %d: %s" % \
               (parent_schema, index, self.error_message)

    def _format_failure_cases(self, failure_cases):
        failure_cases = (
            failure_cases
            .rename("failure_case")
            .reset_index()
            .groupby("failure_case").index.agg([list, len])
            .rename(columns={"list": "index", "len": "count"})
            .sort_values("count", ascending=False)
        )
        self.failure_cases = failure_cases
        if self.n_failure_cases is None:
            return failure_cases
        else:
            return failure_cases.head(self.n_failure_cases)

    def __call__(self, parent_schema, series, index):
        if self.element_wise:
            val_result = series.map(self.fn)
            if val_result.all():
                return True
            raise SchemaError(self.vectorized_error_message(
                parent_schema, index, series[~val_result]))
        else:
            # series-wise validator can return either a boolean or a
            # pd.Series of booleans.
            val_result = self.fn(series)
            if isinstance(val_result, pd.Series):
                if not val_result.dtype == PandasDtype.Bool.value:
                    raise TypeError(
                        "validator %d: %s must return bool or Series of type "
                        "bool, found %s" %
                        (index, self.fn.__name__, val_result.dtype))
                if val_result.all():
                    return True
                try:
                    raise SchemaError(self.vectorized_error_message(
                        parent_schema, index, series[~val_result]))
                except pd.core.indexing.IndexingError:
                    raise SchemaError(
                        self.generic_error_message(parent_schema, index))
            else:
                if val_result:
                    return True
                else:
                    raise SchemaError(
                        self.generic_error_message(parent_schema, index))


class DataFrameSchema(object):
    """A light-weight pandas DataFrame validator."""

    def __init__(self, columns, index=None, transformer=None, coerce=False, strict=False):
        """Initialize pandas dataframe schema.

        Parameters
        ----------
        columns : dict[str -> Column]
            a dict where keys are column names and values are Column objects
            specifying the datatypes and properties of a particular column.
        index : Index
            specify the datatypes and properties of the index.
        transformer : callable
            a callable with signature: pandas.DataFrame -> pandas.DataFrame.
            If specified, calling `validate` will verify properties of the
            columns and return the transformed dataframe object.
        coerce : bool
            whether or not to coerce all of the columns on validation.
        strict : bool
            whether or not to accept columns in the dataframe that aren't in the
            DataFrame Schema.
        """
        self.index = index
        self.columns = columns
        self.transformer = transformer
        self.coerce = coerce
        self.strict = strict

    def validate(self, dataframe):
        # Check if all columns in the dataframe have a corresponding column in
        # the DataFrameSchema
        if self.strict:
            for column in dataframe:
                if column not in self.columns:
                    raise SchemaError(
                        "column '%s' not in DataFrameSchema %s" %
                        (column, self.columns)
                    )

        for c, col in self.columns.items():
            if c not in dataframe and col.required:
                raise SchemaError(
                    "column '%s' not in dataframe\n%s" %
                    (c, dataframe.head()))

            # coercing logic
            if col.coerce or self.coerce:
                dataframe[c] = col.coerce_dtype(dataframe[c])

        schema_elements = [
            col.set_name(col_name) for col_name, col in self.columns.items()
            if col.required or col_name in dataframe
        ]
        if self.index is not None:
            schema_elements += [self.index]
        assert all(s(dataframe) for s in schema_elements)
        if self.transformer is not None:
            dataframe = self.transformer(dataframe)
        return dataframe


class SeriesSchemaBase(object):
    """Base series validator object."""

    def __init__(self, pandas_dtype, checks=None, nullable=False,
                 allow_duplicates=True):
        """Initialize series schema object.

        Parameters
        ----------
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        checks : Check|list[Check]
        nullable : bool
            Whether or not column can contain null values.
        """
        self._pandas_dtype = pandas_dtype
        self._nullable = nullable
        self._allow_duplicates = allow_duplicates
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self._checks = checks

    def __call__(self, series):
        """Validate a series."""
        expected_dtype = _dtype = self._pandas_dtype if \
            isinstance(self._pandas_dtype, str) else self._pandas_dtype.value
        if self._nullable:
            series = series.dropna()
            if _dtype in ["int_", "int8", "int16", "int32", "int64", "uint8",
                          "uint16", "uint32", "uint64"]:
                _dtype = Float.value
                if (series.astype(_dtype) != series).any():
                    # in case where dtype is meant to be int, make sure that
                    # casting to int results in the same values.
                    raise SchemaError(
                        "after dropping null values, expected values in "
                        "series '%s' to be int, found: %s" %
                        (series.name, set(series)))
        else:
            nulls = series.isnull()
            if nulls.sum() > 0:
                type_val_result = series.dtype == _dtype
                if not type_val_result:
                    raise SchemaError(
                        "expected series '%s' to have type %s, got %s and "
                        "non-nullable series contains null values: %s" %
                        (series.name, self._pandas_dtype.value, series.dtype,
                         series[nulls].head(N_FAILURE_CASES).to_dict()))
                else:
                    raise SchemaError(
                        "non-nullable series '%s' contains null values: %s" %
                        (series.name,
                         series[nulls].head(N_FAILURE_CASES).to_dict()))

        # Check if the series contains duplicate values
        if not self._allow_duplicates:
            duplicates = series.duplicated()
            if any(duplicates):
                raise SchemaError(
                    "series '%s' contains duplicate values: %s" %
                    (series.name,
                     series[duplicates].head(N_FAILURE_CASES).to_dict()))

        type_val_result = series.dtype == _dtype
        if not type_val_result:
            raise SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, expected_dtype, series.dtype))

        check_results = []
        for i, check in enumerate(self._checks):
            check_results.append(check(self, series, i))
        return all([type_val_result] + check_results)


class SeriesSchema(SeriesSchemaBase):

    def __init__(self, pandas_dtype, checks=None, nullable=False,
                 allow_duplicates=True):
        """Initialize series schema object.

        Parameters
        ----------
        column : str
            column name in the dataframe
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        checks : callable
            If element_wise is True, then callable signature should be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        nullable : bool
            Whether or not column can contain null values.
        """
        super(SeriesSchema, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates)

    def validate(self, series):
        if not isinstance(series, pd.Series):
            raise TypeError("expected %s, got %s" % (pd.Series, type(series)))
        if super(SeriesSchema, self).__call__(series):
            return series
        raise SchemaError()


class Index(SeriesSchemaBase):

    def __init__(self, pandas_dtype, checks=None, nullable=False,
                 allow_duplicates=True, name=None):
        super(Index, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates)
        self._name = name

    def __call__(self, df):
        return super(Index, self).__call__(pd.Series(df.index))

    def __repr__(self):
        if self._name is None:
            return "<Schema Index>"
        return "<Schema Index: '%s'>" % self._name


class Column(SeriesSchemaBase):

    def __init__(
        self, pandas_dtype, checks=None, nullable=False, allow_duplicates=True,
        coerce=False, required=True
    ):
        """Initialize column validator object.

        Parameters
        ----------
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        checks : callable
            If element_wise is True, then callable signature should be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        nullable : bool
            Whether or not column can contain null values.
        coerce : bool
            Whether or not to coerce the column to the specified pandas_dtype
            before validation
        required: bool
            Whether or not column is allowed to be missing
        """
        super(Column, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates)
        self._name = None
        self.coerce = coerce
        self.required = required

    def set_name(self, name):
        self._name = name
        return self

    def coerce_dtype(self, series):
        _dtype = str if self._pandas_dtype is String \
            else self._pandas_dtype.value
        return series.astype(_dtype)

    def __call__(self, df):
        if self._name is None:
            raise RuntimeError(
                "need to `set_name` of column before calling it.")
        return super(Column, self).__call__(df[self._name])

    def __repr__(self):
        if isinstance(self._pandas_dtype, PandasDtype):
            dtype = self._pandas_dtype.value
        else:
            dtype = self._pandas_dtype
        return "<Schema Column: '%s' type=%s>" % (self._name, dtype)


def check_input(schema, obj_getter=None):
    """Validate function argument when function is called.

    This is a decorator function that validates the schema of a dataframe
    argument in a function. Note that if a transformer is specified by the
    schema, the decorator will return the transformed dataframe, which will be
    passed into the decorated function.

    Parameters
    ----------
    schema : DataFrameSchema|SeriesSchema
        dataframe/series schema object
    obj_getter : int|str|None
        if int, obj_getter refers to the the index of the pandas
        dataframe/series to be validated in the args part of the function
        signature. If str, obj_getter refers to the argument name of the pandas
        dataframe/series in the function signature. This works even if the
        series/dataframe is passed in as a positional argument when the
        function is called. If None, assumes that thedataframe/series is the
        first argument of the decorated function

    """

    @wrapt.decorator
    def _wrapper(fn, instance, args, kwargs):
        args = list(args)
        if isinstance(obj_getter, int):
            args[obj_getter] = schema.validate(args[obj_getter])
        elif isinstance(obj_getter, str):
            if obj_getter in kwargs:
                kwargs[obj_getter] = schema.validate(kwargs[obj_getter])
            else:
                if sys.version_info.major >= 3:
                    arg_spec_args = inspect.getfullargspec(fn).args
                else:
                    arg_spec_args = inspect.getargspec(fn).args
                args_dict = OrderedDict(
                    zip(arg_spec_args, args))
                args_dict[obj_getter] = schema.validate(args_dict[obj_getter])
                args = list(args_dict.values())
        elif obj_getter is None:
            try:
                args[0] = schema.validate(args[0])
            except SchemaError as e:
                raise SchemaError(
                    "error in check_input decorator of function '%s': %s" %
                    (fn.__name__, e))
        else:
            raise ValueError(
                "obj_getter is unrecognized type: %s" % type(obj_getter))
        return fn(*args, **kwargs)

    return _wrapper


def check_output(schema, obj_getter=None):
    """Validate function output.

    Similar to input validator, but validates the output of the decorated
    function. Note that the `transformer` function supplied to the
    DataFrameSchema will not have an effect in the check_output schema
    validator.

    Parameters
    ----------
    schema : DataFrameSchema|SeriesSchema
        dataframe/series schema object
    obj_getter : int|str|callable|None
        if int, assumes that the output of the decorated function is a
        list-like object, where obj_getter is the index of the pandas data
        dataframe/series to be validated. If str, expects that the output
        is a dict-like object, and obj_getter is the key pointing to the
        dataframe/series to be validated. If a callable is supplied, it expects
        the output of decorated function and should return the dataframe/series
        to be validated.

    """

    @wrapt.decorator
    def _wrapper(fn, instance, args, kwargs):
        if schema.transformer is not None:
            warnings.warn(
                "The schema transformer function has no effect in a "
                "check_output decorator. Please perform the necessary "
                "transformations in the '%s' function instead." % fn.__name__)
        out = fn(*args, **kwargs)
        if obj_getter is None:
            obj = out
        elif isinstance(obj_getter, (int, str)):
            obj = out[obj_getter]
        elif callable(obj_getter):
            obj = obj_getter(out)
        else:
            raise ValueError(
                "obj_getter is unrecognized type: %s" % type(obj_getter))
        try:
            schema.validate(obj)
        except SchemaError as e:
            raise SchemaError(
                "error in check_output decorator of function '%s': %s" %
                (fn.__name__, e))

        return out

    return _wrapper
