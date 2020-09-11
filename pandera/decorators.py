"""Decorators for integrating pandera into existing data pipelines."""

import inspect
import warnings

from collections import OrderedDict
from typing import Any, Callable, List, Union, Tuple, Dict, Optional, NoReturn

import pandas as pd

import wrapt

from . import schemas
from . import errors


def _get_fn_argnames(fn: Callable) -> List[str]:
    """Get argument names of a function.

    :param fn: get argument names for this function.
    :returns: list of argument names.
    """
    arg_spec_args = inspect.getfullargspec(fn).args

    if inspect.ismethod(fn) and arg_spec_args[0] == "self":
        # don't include "self" argument
        arg_spec_args = arg_spec_args[1:]
    return arg_spec_args


def _handle_schema_error(
        fn: Callable,
        schema: Union[schemas.DataFrameSchema, schemas.SeriesSchema],
        arg_df: pd.DataFrame,
        schema_error: errors.SchemaError) -> NoReturn:
    """Reraise schema validation error with decorator context.

    :param fn: check the DataFrame or Series input of this function.
    :param schema: dataframe/series schema object
    :param arg_df: dataframe/series we are validating.
    :param schema_error: original exception.
    :raises SchemaError: when ``DataFrame`` violates built-in or custom
        checks.
    """
    msg = (
        "error in check_input decorator of function '%s': %s" %
        (fn.__name__, schema_error)
    )
    raise errors.SchemaError(
        schema, arg_df, msg,
        failure_cases=schema_error.failure_cases,
        check=schema_error.check,
        check_index=schema_error.check_index,
    )


def check_input(
        schema: Union[schemas.DataFrameSchema, schemas.SeriesSchema],
        obj_getter: Optional[Union[str, int]] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False) -> Callable:
    # pylint: disable=duplicate-code
    """Validate function argument when function is called.

    This is a decorator function that validates the schema of a dataframe
    argument in a function. Note that if a transformer is specified by the
    schema, the decorator will return the transformed dataframe, which will be
    passed into the decorated function.

    :param schema: dataframe/series schema object
    :param obj_getter:  (Default value = None) if int, obj_getter refers to the
        the index of the pandas dataframe/series to be validated in the args
        part of the function signature. If str, obj_getter refers to the
        argument name of the pandas dataframe/series in the function signature.
        This works even if the series/dataframe is passed in as a positional
        argument when the function is called. If None, assumes that the
        dataframe/series is the first argument of the decorated function
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
    :returns: wrapped function

    :example:

    Check the input of a decorated function.

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> schema = pa.DataFrameSchema({"column": pa.Column(pa.Int)})
    >>>
    >>> @pa.check_input(schema)
    ... def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ...     df["doubled_column"] = df["column"] * 2
    ...     return df
    >>>
    >>> df = pd.DataFrame({
    ...     "column": range(5),
    ... })
    >>>
    >>> transform_data(df)
       column  doubled_column
    0       0               0
    1       1               2
    2       2               4
    3       3               6
    4       4               8

    See :ref:`here<decorators>` for more usage details.

    """

    @wrapt.decorator
    def _wrapper(
            fn: Callable,
            instance: Union[None, Any],
            args: Union[List[Any], Tuple[Any]],
            kwargs: Dict[str, Any]):
        # pylint: disable=unused-argument
        """Check pandas DataFrame or Series before calling the function.

        :param fn: check the DataFrame or Series input of this function
        :param instance: the object to which the wrapped function was bound
            when it was called. Only applies to methods.
        :param args: the list of positional arguments supplied when the
            decorated function was called.
        :param kwargs: the dictionary of keyword arguments supplied when the
            decorated function was called.
        """
        args = list(args)
        validate_args = (head, tail, sample, random_state, lazy)
        if isinstance(obj_getter, int):
            try:
                args[obj_getter] = schema.validate(args[obj_getter])
            except IndexError as exc:
                raise IndexError(
                    "error in check_input decorator of function '%s': the "
                    "index '%s' was supplied to the check but this "
                    "function accepts '%s' arguments, so the maximum "
                    "index is '%s'. The full error is: '%s'" %
                    (
                        fn.__name__,
                        obj_getter,
                        len(_get_fn_argnames(fn)),
                        max(0, len(_get_fn_argnames(fn)) - 1),
                        exc
                    )
                ) from exc
        elif isinstance(obj_getter, str):
            if obj_getter in kwargs:
                kwargs[obj_getter] = schema.validate(
                    kwargs[obj_getter], *validate_args
                )
            else:
                arg_spec_args = _get_fn_argnames(fn)
                args_dict = OrderedDict(
                    zip(arg_spec_args, args))
                args_dict[obj_getter] = schema.validate(
                    args_dict[obj_getter], *validate_args)
                args = list(args_dict.values())
        elif obj_getter is None and args:
            try:
                args[0] = schema.validate(args[0], *validate_args)
            except errors.SchemaError as e:
                _handle_schema_error(fn, schema, args[0], e)
        elif obj_getter is None and kwargs:
            # get the first key in the same order specified in the
            # function argument.
            args_names = _get_fn_argnames(fn)

            try:
                kwargs[args_names[0]] = schema.validate(
                    kwargs[args_names[0]], *validate_args
                )
            except errors.SchemaError as e:
                _handle_schema_error(fn, schema, kwargs[args_names[0]], e)
        else:
            raise ValueError(
                "obj_getter is unrecognized type: %s" % type(obj_getter))
        return fn(*args, **kwargs)

    return _wrapper


def check_output(
        schema: Union[schemas.DataFrameSchema, schemas.SeriesSchema],
        obj_getter: Optional[Union[int, str, Callable]] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False) -> Callable:
    # pylint: disable=duplicate-code
    """Validate function output.

    Similar to input validator, but validates the output of the decorated
    function. Note that the `transformer` function supplied to the
    DataFrameSchema will not have an effect in the check_output schema
    validator.

    :param schema: dataframe/series schema object
    :param obj_getter:  (Default value = None) if int, assumes that the output
        of the decorated function is a list-like object, where obj_getter is
        the index of the pandas data dataframe/series to be validated. If str,
        expects that the output is a dict-like object, and obj_getter is the
        key pointing to the dataframe/series to be validated. If a callable is
        supplied, it expects the output of decorated function and should return
        the dataframe/series to be validated.
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
    :returns: wrapped function

    :example:

    Check the output a decorated function.

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> schema = pa.DataFrameSchema(
    ...     columns={"doubled_column": pa.Column(pa.Int)},
    ...     checks=pa.Check(
    ...         lambda df: df["doubled_column"] == df["column"] * 2
    ...     )
    ... )
    >>>
    >>> @pa.check_output(schema)
    ... def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ...     df["doubled_column"] = df["column"] * 2
    ...     return df
    >>>
    >>> df = pd.DataFrame({"column": range(5)})
    >>>
    >>> transform_data(df)
       column  doubled_column
    0       0               0
    1       1               2
    2       2               4
    3       3               6
    4       4               8

    See :ref:`here<decorators>` for more usage details.
    """

    @wrapt.decorator
    def _wrapper(
            fn: Callable,
            instance: Union[None, Any],
            args: Union[List[Any], Tuple[Any]],
            kwargs: Dict[str, Any]):
        # pylint: disable=unused-argument
        """Check pandas DataFrame or Series before calling the function.

        :param fn: check the DataFrame or Series output of this function
        :param instance: the object to which the wrapped function was bound
            when it was called. Only applies to methods.
        :param args: the list of positional arguments supplied when the
            decorated function was called.
        :param kwargs: the dictionary of keyword arguments supplied when the
            decorated function was called.
        """
        if hasattr(schema, "transformer") and \
                getattr(schema, "transformer") is not None:
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
            schema.validate(obj, head, tail, sample, random_state, lazy)
        except errors.SchemaError as e:
            msg = (
                "error in check_output decorator of function '%s': %s" %
                (fn.__name__, e)
            )
            raise errors.SchemaError(
                schema, obj, msg,
                failure_cases=e.failure_cases,
                check=e.check,
                check_index=e.check_index,
            )

        return out

    return _wrapper
