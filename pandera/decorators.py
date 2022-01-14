"""Decorators for integrating pandera into existing data pipelines."""
import functools
import inspect
import sys
import typing
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import pandas as pd
import wrapt
from pydantic import validate_arguments

from . import errors, schemas
from .inspection_utils import (
    is_classmethod_from_meta,
    is_decorated_classmethod,
)
from .model import SchemaModel
from .typing import AnnotationInfo

Schemas = Union[schemas.DataFrameSchema, schemas.SeriesSchema]
InputGetter = Union[str, int]
OutputGetter = Union[str, int, Callable]
F = TypeVar("F", bound=Callable)


def _get_fn_argnames(fn: Callable) -> List[str]:
    """Get argument names of a function.

    :param fn: get argument names for this function.
    :returns: list of argument names to be matched with the positional
    args passed in the decorator.

    .. note::
       Excludes first positional "self" or "cls" arguments if needed:
       - exclude self:
           - if fn is a method (self being an implicit argument)
       - exclude cls:
           - if fn is a decorated classmethod in Python 3.9+
           - if fn is declared as a regular method on a metaclass

    For functions decorated with ``@classmethod``, cls is excluded only in Python 3.9+
    because that is when Python's handling of classmethods changed and wrapt mirrors it.
    See: https://github.com/GrahamDumpleton/wrapt/issues/182
    """
    arg_spec_args = inspect.getfullargspec(fn).args
    first_arg_is_self = arg_spec_args[0] == "self"
    is_py_newer_than_39 = sys.version_info[:2] >= (3, 9)
    # Exclusion criteria
    is_regular_method = inspect.ismethod(fn) and first_arg_is_self
    is_decorated_cls_method = (
        is_decorated_classmethod(fn) and is_py_newer_than_39
    )
    is_cls_method_from_meta_method = is_classmethod_from_meta(fn)
    if (
        is_regular_method
        or is_decorated_cls_method
        or is_cls_method_from_meta_method
    ):
        # Don't include "self" / "cls" argument
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
    msg = f"error in {decorator_name} decorator of function '{fn.__name__}': {schema_error}"
    raise errors.SchemaError(
        schema,
        arg_df,
        msg,
        failure_cases=schema_error.failure_cases,
        check=schema_error.check,
        check_index=schema_error.check_index,
    ) from schema_error


def check_input(
    schema: Schemas,
    obj_getter: Optional[InputGetter] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable[[F], F]:
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
    >>> schema = pa.DataFrameSchema({"column": pa.Column(int)})
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
        args: Tuple[Any, ...],
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
                    f"error in check_input decorator of function '{fn.__name__}': the "
                    f"index '{obj_getter}' was supplied to the check but this "
                    f"function accepts '{len(_get_fn_argnames(fn))}' arguments, so the maximum "
                    f"index is 'max(0, len(_get_fn_argnames(fn)) - 1)'. The full error is: '{exc}'"
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
) -> Callable[[F], F]:
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
    ...     columns={"doubled_column": pa.Column(int)},
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

    def validate(out: Any, fn: Callable) -> None:
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

    @wrapt.decorator
    def _wrapper(
        fn: Callable,
        instance: Union[None, Any],
        args: Tuple[Any, ...],
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
        if inspect.iscoroutinefunction(fn):

            async def aio_wrapper():
                res = await fn(*args, **kwargs)
                validate(res, fn)
                return res

            return aio_wrapper()
        else:
            out = fn(*args, **kwargs)
            validate(out, fn)
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
    **inputs: Schemas,
) -> Callable[[F], F]:
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
        args: Tuple[Any, ...],
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


@overload
def check_types(
    wrapped: F,
    *,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> F:
    ...  # pragma: no cover


@overload
def check_types(
    wrapped: None = None,
    *,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable[[F], F]:
    ...  # pragma: no cover


def check_types(
    wrapped=None,
    *,
    with_pydantic=False,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable:
    """Validate function inputs and output based on type annotations.

    See the :ref:`User Guide <schema_models>` for more.

    :param wrapped: the function to decorate.
    :param with_pydantic: use ``pydantic.validate_arguments`` to validate
        inputs. This function is still needed to validate function outputs.
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
    # pylint: disable=too-many-locals
    if wrapped is None:
        return functools.partial(
            check_types,
            with_pydantic=with_pydantic,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    # Front-load annotation parsing
    annotated_schema_models: Dict[str, Tuple[SchemaModel, AnnotationInfo]] = {}
    for arg_name_, annotation in typing.get_type_hints(wrapped).items():
        annotation_info = AnnotationInfo(annotation)
        if not annotation_info.is_generic_df:
            continue

        schema_model = cast(SchemaModel, annotation_info.arg)
        annotated_schema_models[arg_name_] = (schema_model, annotation_info)

    def _check_arg(arg_name: str, arg_value: Any) -> Any:
        """
        Validate function's argument if annoted with a schema, else
        pass-through.
        """
        schema_model, annotation_info = annotated_schema_models.get(
            arg_name, (None, None)
        )

        if schema_model is None:
            return arg_value

        if (
            annotation_info
            and not (annotation_info.optional and arg_value is None)
            # the pandera.schema attribute should only be available when
            # schema.validate has been called in the DF. There's probably
            # a better way of doing this
        ):
            config = schema_model.__config__
            data_container_type = annotation_info.origin
            schema = schema_model.to_schema()

            if data_container_type and config and config.from_format:
                arg_value = data_container_type.from_format(arg_value, config)

            if (
                arg_value.pandera.schema is None
                # don't re-validate a dataframe that contains the same exact
                # schema
                or arg_value.pandera.schema != schema
            ):
                try:
                    arg_value = schema.validate(
                        arg_value,
                        head,
                        tail,
                        sample,
                        random_state,
                        lazy,
                        inplace,
                    )
                except errors.SchemaError as e:
                    _handle_schema_error(
                        "check_types", wrapped, schema, arg_value, e
                    )

            if data_container_type and config and config.to_format:
                arg_value = data_container_type.to_format(arg_value, config)

            return arg_value

    sig = inspect.signature(wrapped)

    def validate_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            arg_name: _check_arg(arg_name, arg_value)
            for arg_name, arg_value in arguments.items()
        }

    def validate_inputs(
        instance: Optional[Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if instance is not None:
            # If the wrapped function is a method -> add "self" as the first positional arg
            args = (instance, *args)

        validated_pos = validate_args(sig.bind_partial(*args).arguments)
        validated_kwd = validate_args(sig.bind_partial(**kwargs).arguments)

        if instance is not None:
            # If the decorated func is a method, "wrapped" is a bound method
            # -> remove "self" before passing positional args through
            first_pos_arg = list(sig.parameters)[0]
            del validated_pos[first_pos_arg]

        return validated_pos, validated_kwd

    if inspect.iscoroutinefunction(wrapped):

        @wrapt.decorator
        async def _wrapper(
            wrapped_: Callable,
            instance: Optional[Any],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ):
            if with_pydantic:
                out = await validate_arguments(wrapped_)(*args, **kwargs)
            else:
                validated_pos, validated_kwd = validate_inputs(
                    instance, args, kwargs
                )
                out = await wrapped_(*validated_pos.values(), **validated_kwd)
            return _check_arg("return", out)

    else:

        @wrapt.decorator
        def _wrapper(
            wrapped_: Callable,
            instance: Optional[Any],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ):
            if with_pydantic:
                out = validate_arguments(wrapped_)(*args, **kwargs)
            else:
                validated_pos, validated_kwd = validate_inputs(
                    instance, args, kwargs
                )
                out = wrapped_(*validated_pos.values(), **validated_kwd)
            return _check_arg("return", out)

    return _wrapper(wrapped)  # pylint:disable=no-value-for-parameter
