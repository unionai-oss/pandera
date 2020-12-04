"""Decorators for integrating pandera into existing data pipelines."""

import functools
import inspect
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
    cast,
)

import pandas as pd
import wrapt

from . import errors, schemas
from .model import SchemaModel
from .typing import AnnotationInfo

Schemas = Union[schemas.DataFrameSchema, schemas.SeriesSchema]
InputGetter = Union[str, int]
OutputGetter = Union[str, int, Callable]


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
    decorator_name,
    fn: Callable,
    schema: Union[schemas.DataFrameSchema, schemas.SeriesSchema],
    arg_df: pd.DataFrame,
    schema_error: errors.SchemaError,
) -> NoReturn:
    """Reraise schema validation error with decorator context.

    :param fn: check the DataFrame or Series input of this function.
    :param schema: dataframe/series schema object
    :param arg_df: dataframe/series we are validating.
    :param schema_error: original exception.
    :raises SchemaError: when ``DataFrame`` violates built-in or custom
        checks.
    """
    msg = "error in %s decorator of function '%s': %s" % (
        decorator_name,
        fn.__name__,
        schema_error,
    )
    raise errors.SchemaError(
        schema,
        arg_df,
        msg,
        failure_cases=schema_error.failure_cases,
        check=schema_error.check,
        check_index=schema_error.check_index,
    )


def check_input(
    schema: Schemas,
    obj_getter: Optional[InputGetter] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable:
    # pylint: disable=duplicate-code
    """Validate function argument when function is called.

    This is a decorator function that validates the schema of a dataframe
    argument in a function.

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
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
        otherwise creates a copy of the data.
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
        kwargs: Dict[str, Any],
    ):
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
        validate_args = (head, tail, sample, random_state, lazy, inplace)
        if isinstance(obj_getter, int):
            try:
                args[obj_getter] = schema.validate(args[obj_getter])
            except IndexError as exc:
                raise IndexError(
                    "error in check_input decorator of function '%s': the "
                    "index '%s' was supplied to the check but this "
                    "function accepts '%s' arguments, so the maximum "
                    "index is '%s'. The full error is: '%s'"
                    % (
                        fn.__name__,
                        obj_getter,
                        len(_get_fn_argnames(fn)),
                        max(0, len(_get_fn_argnames(fn)) - 1),
                        exc,
                    )
                ) from exc
        elif isinstance(obj_getter, str):
            if obj_getter in kwargs:
                kwargs[obj_getter] = schema.validate(
                    kwargs[obj_getter], *validate_args
                )
            else:
                arg_spec_args = _get_fn_argnames(fn)
                args_dict = OrderedDict(zip(arg_spec_args, args))
                args_dict[obj_getter] = schema.validate(
                    args_dict[obj_getter], *validate_args
                )
                args = list(args_dict.values())
        elif obj_getter is None and args:
            try:
                args[0] = schema.validate(args[0], *validate_args)
            except errors.SchemaError as e:
                _handle_schema_error("check_input", fn, schema, args[0], e)
        elif obj_getter is None and kwargs:
            # get the first key in the same order specified in the
            # function argument.
            args_names = _get_fn_argnames(fn)

            try:
                kwargs[args_names[0]] = schema.validate(
                    kwargs[args_names[0]], *validate_args
                )
            except errors.SchemaError as e:
                _handle_schema_error(
                    "check_input", fn, schema, kwargs[args_names[0]], e
                )
        else:
            raise TypeError(
                f"obj_getter is unrecognized type: {type(obj_getter)}"
            )
        return fn(*args, **kwargs)

    return _wrapper


def check_output(
    schema: Schemas,
    obj_getter: Optional[OutputGetter] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable:
    # pylint: disable=duplicate-code
    """Validate function output.

    Similar to input validator, but validates the output of the decorated
    function.

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
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
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
        kwargs: Dict[str, Any],
    ):
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
        out = fn(*args, **kwargs)
        if obj_getter is None:
            obj = out
        elif isinstance(obj_getter, (int, str)):
            obj = out[obj_getter]
        elif callable(obj_getter):
            obj = obj_getter(out)
        else:
            raise TypeError(
                f"obj_getter is unrecognized type: {type(obj_getter)}"
            )
        try:
            schema.validate(
                obj, head, tail, sample, random_state, lazy, inplace
            )
        except errors.SchemaError as e:
            _handle_schema_error("check_output", fn, schema, obj, e)

        return out

    return _wrapper


def check_io(
    head: int = None,
    tail: int = None,
    sample: int = None,
    random_state: int = None,
    lazy: bool = False,
    inplace: bool = False,
    out: Union[
        Schemas,
        Tuple[OutputGetter, Schemas],
        List[Tuple[OutputGetter, Schemas]],
    ] = None,
    **inputs: Dict[InputGetter, Schemas],
) -> Callable:
    """Check schema for multiple inputs and outputs.

    See :ref:`here<decorators>` for more usage details.

    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    :param random_state: random seed for the ``sample`` argument.
    :param lazy: if True, lazily evaluates dataframe against all validation
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
        otherwise creates a copy of the data.
    :param out: this should be a schema object if the function outputs a single
        dataframe/series. It can be a two-tuple, where the first element is
        a string, integer, or callable that fetches the pandas data structure
        in the output, and the second element is the schema to validate
        against. For multiple outputs, specify a list of two-tuples following
        the above structure.
    :param inputs: kwargs keys should be the argument name in the decorated
        function and values should be the schema used to validate the pandas
        data structure referenced by the argument name.
    :returns: wrapped function
    """
    check_args = (head, tail, sample, random_state, lazy, inplace)

    @wrapt.decorator
    def _wrapper(
        fn: Callable,
        instance: Union[None, Any],  # pylint: disable=unused-argument
        args: Union[List[Any], Tuple[Any]],
        kwargs: Dict[str, Any],
    ):
        """Check pandas DataFrame or Series before calling the function.

        :param fn: check the DataFrame or Series output of this function
        :param instance: the object to which the wrapped function was bound
            when it was called. Only applies to methods.
        :param args: the list of positional arguments supplied when the
            decorated function was called.
        :param kwargs: the dictionary of keyword arguments supplied when the
            decorated function was called.
        """
        out_schemas = out
        if isinstance(out, list):
            out_schemas = out
        elif isinstance(out, (schemas.DataFrameSchema, schemas.SeriesSchema)):
            out_schemas = [(None, out)]  # type: ignore
        elif isinstance(out, tuple):
            out_schemas = [out]
        elif out is None:
            out_schemas = []
        else:
            raise TypeError(
                f"type of out argument not recognized: {type(out)}"
            )

        wrapped_fn = fn
        for input_getter, input_schema in inputs.items():
            # pylint: disable=no-value-for-parameter
            wrapped_fn = check_input(
                input_schema, input_getter, *check_args  # type: ignore
            )(wrapped_fn)

        # pylint: disable=no-value-for-parameter
        for out_getter, out_schema in out_schemas:  # type: ignore
            wrapped_fn = check_output(out_schema, out_getter, *check_args)(
                wrapped_fn
            )

        return wrapped_fn(*args, **kwargs)

    return _wrapper


def check_types(
    wrapped=None,
    *,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable:
    """Validate function inputs and output based on type annotations.

    See the :ref:`User Guide <schema_models>` for more.

    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    :param random_state: random seed for the ``sample`` argument.
    :param lazy: if True, lazily evaluates dataframe against all validation
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
    """
    if wrapped is None:
        return functools.partial(
            check_types,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @wrapt.decorator
    def _wrapper(
        wrapped: Callable,
        instance: Optional[Any],  # pylint:disable=unused-argument
        args: Union[List[Any], Tuple[Any]],
        kwargs: Dict[str, Any],
    ):
        sig = inspect.signature(wrapped)

        arguments = sig.bind(*args, **kwargs).arguments
        for arg_name, arg_value in arguments.items():
            annotation = sig.parameters[arg_name].annotation
            annotation_info = AnnotationInfo(annotation)

            if annotation_info.optional and arg_value is None:
                continue

            if not annotation_info.is_generic_df:
                continue

            model = cast(SchemaModel, annotation_info.arg)
            schema = model.to_schema()
            try:
                schema.validate(
                    arg_value, head, tail, sample, random_state, lazy, inplace
                )
            except errors.SchemaError as e:
                _handle_schema_error(
                    "check_types", wrapped, schema, arg_value, e
                )

        out = wrapped(*args, **kwargs)

        annotation_info = AnnotationInfo(sig.return_annotation)
        if annotation_info.optional and out is None:
            return out

        if annotation_info.is_generic_df:
            model = cast(SchemaModel, annotation_info.arg)
            schema = model.to_schema()
            try:
                schema.validate(
                    out, head, tail, sample, random_state, lazy, inplace
                )
            except errors.SchemaError as e:
                _handle_schema_error("check_types", wrapped, schema, out, e)

        return out

    return _wrapper(wrapped)  # pylint:disable=no-value-for-parameter
