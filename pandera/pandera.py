"""Validate Pandas Data Structures."""

import inspect
import pandas as pd
import wrapt

from collections import OrderedDict
from enum import Enum
from schema import Schema, Use, And, SchemaError


class PandasDtype(Enum):
    Bool = "bool"
    DateTime = "datetime64[ns]"
    Category = "category"
    Float = "float"
    Int = "int"
    Object = "object"
    String = "object"
    Timedelta = "timedelta64[ns]"


class DataFrameSchema(object):
    """A light-weight pandas DataFrame validator."""

    def __init__(self, columns, transformer=None):
        """Initialize pandas dataframe schema.

        Parameters
        ----------
        columns : list of Column
            a list of Column objects specifying the datatypes and properties
            of a particular column.
        transformer : callable
            a callable with signature: pandas.DataFrame -> pandas.DataFrame.
            If specified, calling `validate` will verify properties of the
            columns and return the transformed dataframe object.

        """
        self.transformer = transformer
        schema_arg = And(*columns)
        if self.transformer is not None:
            schema_arg = And(schema_arg, Use(transformer))
        self.schema = Schema(schema_arg)

    def validate(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("expected series, got %s" % type(dataframe))
        return self.schema.validate(dataframe)


class SeriesSchemaBase(object):
    """Column validator object."""

    def __init__(self, pandas_dtype, validators=None, element_wise=True):
        """Initialize column validator object.

        Parameters
        ----------
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        validators : callable|list[callable]
            list of function with which to check series schema. If element_wise
            is True, then callable signature should be: x -> x where x is a
            scalar element in the column. Otherwise, x is assumed to be a
            pandas.Series object. Can also be a single callable.
        element_wise : bool|list[bool]
            Whether or not to apply validator in an element-wise fashion. If
            bool, assumes that all validators should be applied to the column
            element-wise. If list, should be the same number of elements
            as validators.
        """
        self._pandas_dtype = pandas_dtype
        if validators is None:
            validators = []
        if callable(validators):
            validators = [validators]
        if isinstance(element_wise, bool):
            element_wise = [element_wise] * len(validators)
        if len(validators) != len(element_wise):
            raise ValueError(
                "validators and element_wise must be the same length")
        self._element_wise = element_wise
        self._validators = validators

    def __call__(self, series):
        """Validate a series."""
        _dtype = self._pandas_dtype if isinstance(self._pandas_dtype, str) \
            else self._pandas_dtype.value
        type_val_result = series.dtype == _dtype
        if not type_val_result:
            raise SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, self._pandas_dtype.value, series.dtype))
        for validator, element_wise in zip(
                self._validators, self._element_wise):
            self._validate(series, validator, element_wise)
        return type_val_result

    def _validate(self, series, validator, element_wise):
        validator_schema = Schema(validator)
        if element_wise:
            try:
                series.map(validator_schema.validate)
            except SchemaError:
                failure_cases = ~series.map(validator)
                raise SchemaError(
                    "series did not pass element-wise validator "
                    "'%s'. failure cases: %s" %
                    (inspect.getsource(validator).strip(),
                     series[failure_cases].to_dict()))
        else:
            try:
                validator_schema.validate(series)
            except SchemaError:
                raise SchemaError(
                    "series did not pass series validator '%s', " %
                    (inspect.getsource(validator).strip()))


class SeriesSchema(SeriesSchemaBase):

    def __init__(self, pandas_dtype, validators=None, element_wise=True):
        super(SeriesSchema, self).__init__(
            pandas_dtype, validators, element_wise)

    def validate(self, series):
        if not isinstance(series, pd.Series):
            raise TypeError("expected %s, got %s" % (pd.Series, type(series)))
        return super(SeriesSchema, self).__call__(series)


class Column(SeriesSchemaBase):

    def __init__(
            self, column, pandas_dtype, validators=None, element_wise=True):
        """Initialize column validator object.

        Parameters
        ----------
        column : str
            column name in the dataframe
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        validators : callable
            If element_wise is True, then callable signature should be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        element_wise : bool
            Whether or not to apply validator in an element-wise fashion.
        """
        super(Column, self).__init__(pandas_dtype, validators, element_wise)
        self._column = column

    def __call__(self, df):
        return super(Column, self).__call__(df[self._column])

    def __repr__(self):
        return "<Column: %s>" % self._column


def validate_input(schema, obj_getter=None):
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


def validate_output(schema, obj_getter=None):
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
