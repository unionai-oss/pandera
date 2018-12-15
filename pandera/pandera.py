"""Validate Pandas Data Structures."""

import inspect
import pandas as pd
import wrapt

from collections import OrderedDict
from enum import Enum
from functools import partial
from schema import Schema, Use, And, SchemaError


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


class Check(object):

    def __init__(self, fn, element_wise=True, error=None):
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

        """
        self.fn = fn
        self.element_wise = element_wise
        self.error = error

    @property
    def error_message(self):
        if self.error:
            return "%s: %s" % (self.fn.__name__, self.error)
        return "%s" % self.fn.__name__

    def vectorized_error_message(self, parent_schema, index, failure_cases):
        return (
            "%s failed element-wise validator %d:\n"
            "%s\nfailure cases: %s" %
            (parent_schema, index, self.error_message, failure_cases))

    def __call__(self, parent_schema, series, index):
        if self.element_wise:
            val_result = series.map(self.fn)
            if val_result.all():
                return True
            raise SchemaError(self.vectorized_error_message(
                parent_schema, index, series[~val_result].to_dict()))
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
                raise SchemaError(self.vectorized_error_message(
                    parent_schema, index, series[~val_result].to_dict()))
            else:
                if val_result:
                    return True
                else:
                    raise SchemaError(
                        "series did not pass series validator %d: %s" %
                        (index, self.error_message))


class DataFrameSchema(object):
    """A light-weight pandas DataFrame validator."""

    def __init__(self, columns, index=None, transformer=None):
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

        """
        self.index = index
        self.columns = columns
        self.transformer = transformer

    def validate(self, dataframe):
        for c in self.columns:
            if c not in dataframe:
                raise SchemaError(
                    "column '%s' not in dataframe\n%s" %
                    (c, dataframe.head()))

        schema_arg = [
            partial(col, col_name) for col_name, col in self.columns.items()]
        if self.index is not None:
            schema_arg += [self.index]
        schema_arg = And(*schema_arg)
        if self.transformer is not None:
            schema_arg = And(schema_arg, Use(self.transformer))
        self.schema = Schema(schema_arg)

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("expected dataframe, got %s" % type(dataframe))
        return self.schema.validate(dataframe)


class SeriesSchemaBase(object):
    """Base series validator object."""

    def __init__(self, pandas_dtype, checks=None, nullable=False):
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
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self._checks = checks

    def __call__(self, series):
        """Validate a series."""
        _dtype = self._pandas_dtype if isinstance(self._pandas_dtype, str) \
            else self._pandas_dtype.value
        if self._nullable:
            series = series.dropna()
            _series = series.astype(_dtype)
            # in case where dtype is meant to be int, make sure that casting
            # to int results in the same values.
            if (_dtype == "int64") and (_series != series).any():
                raise SchemaError(
                    "after dropping null values, expected series values to "
                    "be int, found: %s" % set(series))
            series = _series
        else:
            nulls = series.isnull()
            if nulls.sum() > 0:
                raise SchemaError(
                    "non-nullable series contains null values: %s" %
                    series[nulls].to_dict())

        type_val_result = series.dtype == _dtype
        if not type_val_result:
            raise SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, self._pandas_dtype.value, series.dtype))
        check_results = []
        for i, validator in enumerate(self._checks):
            check_results.append(validator(self, series, i))
        return all([type_val_result] + check_results)


class SeriesSchema(SeriesSchemaBase):

    def __init__(self, pandas_dtype, checks=None, nullable=False):
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
        super(SeriesSchema, self).__init__(pandas_dtype, checks, nullable)

    def validate(self, series):
        if not isinstance(series, pd.Series):
            raise TypeError("expected %s, got %s" % (pd.Series, type(series)))
        if super(SeriesSchema, self).__call__(series):
            return series
        raise SchemaError()


class Index(SeriesSchemaBase):

    def __init__(self, pandas_dtype, checks=None, nullable=False,
                 name=None):
        super(Index, self).__init__(pandas_dtype, checks, nullable)
        self._name = name

    def __call__(self, df):
        return super(Index, self).__call__(pd.Series(df.index))

    def __repr__(self):
        if self._name is None:
            return "<Schema Index>"
        return "<Schema Index: %s>" % self._name


class Column(SeriesSchemaBase):

    def __init__(
            self, pandas_dtype, checks=None, nullable=False):
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
        """
        super(Column, self).__init__(pandas_dtype, checks, nullable)

    def __call__(self, column_name, df):
        return super(Column, self).__call__(df[column_name])

    def __repr__(self):
        if isinstance(self._pandas_dtype, PandasDtype):
            dtype = self._pandas_dtype.value
        else:
            dtype = self._pandas_dtype
        return "<Schema Column: type=%s>" % dtype


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
                args_dict = OrderedDict(zip(inspect.getargspec(fn).args, args))
                args_dict[obj_getter] = schema.validate(args_dict[obj_getter])
                args = list(args_dict.values())
        elif obj_getter is None:
            args[0] = schema.validate(args[0])
        else:
            raise ValueError(
                "obj_getter is unrecognized type: %s" % type(obj_getter))
        return fn(*args, **kwargs)
    return _wrapper


def check_output(schema, obj_getter=None):
    """Validate function output.

    Similar to input validator, but validates the output of the decorated
    function.

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
        out = fn(*args, **kwargs)
        if isinstance(obj_getter, int):
            obj = out[obj_getter]
        elif isinstance(obj_getter, str):
            obj = out[obj_getter]
        elif callable(obj_getter):
            obj = obj_getter(out)
        elif obj_getter is None:
            obj = out
        schema.validate(obj)
        return out
    return _wrapper
