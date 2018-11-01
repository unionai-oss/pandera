"""Validate Pandas Data Structures."""

import functools
import inspect

from collections import OrderedDict
from enum import Enum
from schema import Schema, Use, And, SchemaError


class PandasDtype(Enum):
    Bool = "bool"
    DateTime = "datetime64[ns]"
    DateTimeTZ = "datetime64[ns, tz]"
    Category = "category"
    Float = "float"
    Int = "int"
    Object = "object"
    String = "object"
    Timedelta = "timedelta[ns]"


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
        return self.schema.validate(dataframe)


class SeriesSchemaBase(object):
    """Column validator object."""

    def __init__(
            self, pandas_dtype, validator=None, element_wise=True):
        """Initialize column validator object.

        Parameters
        ----------
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        validator : callable
            If element_wise is True, then callable signature should be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        element_wise : bool
            Whether or not to apply validator in an element-wise fashion.
        """
        self._pandas_dtype = pandas_dtype
        self._validator = validator
        self._element_wise = element_wise

    def __call__(self, series):
        _dtype = self._pandas_dtype if isinstance(self._pandas_dtype, str) \
            else self._pandas_dtype.value
        type_val_result = series.dtype == _dtype
        if not type_val_result:
            raise SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, self._pandas_dtype.value, series.dtype))
        if self._validator is not None:
            validator_schema = Schema(self._validator)
            if self._element_wise:
                try:
                    series.map(validator_schema.validate)
                except SchemaError:
                    failure_cases = ~series.map(self._validator)
                    raise SchemaError(
                        "series '%s' did not pass element-wise validator "
                        "'%s'. failure cases: %s" %
                        (series.name, inspect.getsource(
                            self._validator).strip(),
                         series[failure_cases].tolist()))
            else:
                try:
                    validator_schema.validate(series)
                except SchemaError:
                    raise SchemaError(
                        "series '%s' did not pass series validator '%s', " %
                        (series.name,
                         inspect.getsource(self._validator).strip()))
        return type_val_result


class SeriesSchema(SeriesSchemaBase):

    def __init__(self, pandas_dtype, validator=None, element_wise=True):
        super(SeriesSchema, self).__init__(
            pandas_dtype, validator=None, element_wise=True)

    def validate(self, series):
        super(SeriesSchema, self).__call__(series)


class Column(SeriesSchemaBase):

    def __init__(
            self, column, pandas_dtype, validator=None, element_wise=True):
        """Initialize column validator object.

        Parameters
        ----------
        column : str
            column name in the dataframe
        pandas_dtype : str|PandasDtype
            datatype of the column. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        validator : callable
            If element_wise is True, then callable signature should be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        element_wise : bool
            Whether or not to apply validator in an element-wise fashion.
        """
        super(Column, self).__init__(pandas_dtype, validator, element_wise)
        self._column = column

    def __call__(self, df):
        return super(Column, self).__call__(df[self._column])

    def __repr__(self):
        return "<Column: %s>" % self._column


def validate_dataframe_arg(df_arg, schema, is_positional=True):
    """Validate function argument when function is called.

    This is a decorator function that validates the schema of a dataframe
    argument in a function. Note that if a transformer is specified by the
    schema, the decorator will return the transformed dataframe, which will be
    passed into the decorated function.

    Parameters
    ----------
    df_arg : str
        name of the dataframe argument in the function being decorated
    schema : PandasDataFrameSchema
        dataframe schema object
    is_positional : bool
        whether or not the dataframe argument is a positional or keyword
        argument. Default = True.
    """
    def _validator(fn):

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            if is_positional:
                arg_dict = OrderedDict(zip(inspect.getargspec(fn).args, args))
            else:
                arg_dict = kwargs
            df = schema.validate(arg_dict[df_arg])
            arg_dict[df_arg] = df
            if is_positional:
                fn(*list(arg_dict.values()), **kwargs)
            else:
                fn(*args, **arg_dict)

        return _wrapper

    return _validator
