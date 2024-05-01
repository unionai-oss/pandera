"""This module holds the decorators only valid for pyspark"""

import functools
import logging
import warnings
from contextlib import contextmanager
from typing import List, Type

from pyspark.sql import DataFrame

from pandera.api.pyspark.types import PysparkDefaultTypes
from pandera.config import ValidationDepth, get_config_context
from pandera.errors import SchemaError
from pandera.validation_depth import ValidationScope

logger = logging.getLogger(__name__)


def register_input_datatypes(
    acceptable_datatypes: List[Type[PysparkDefaultTypes]] = None,
):
    """
    This decorator is used to register the input datatype for the check.
    An Error would br raised in case the type is not in the list of acceptable types.

    :param acceptable_datatypes: List of pyspark datatypes for which the function is applicable
    """

    def wrapper(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            # Get the pyspark object from arguments
            pyspark_object = list(args)[0]
            validation_df = pyspark_object.dataframe
            validation_column = pyspark_object.column_name
            pandera_schema_datatype = validation_df.pandera.schema.get_dtypes(
                validation_df
            )[validation_column].type.typeName
            # Type Name of the valid datatypes needed for comparison  to remove the parameterized values since
            # only checking type not the parameters
            valid_datatypes = [i.typeName for i in acceptable_datatypes]
            current_datatype = (
                validation_df.select(validation_column)
                .schema[0]
                .dataType.typeName
            )
            if pandera_schema_datatype != current_datatype:
                raise SchemaError(
                    schema=validation_df.pandera.schema,
                    data=validation_df,
                    message=f'The check with name "{func.__name__}" was expected to be run for \n'
                    f"{pandera_schema_datatype()} but got {current_datatype()} instead from the input. \n"
                    f"This error is usually caused by schema mismatch the value is different from schema defined in "
                    f"pandera schema and one in the dataframe",
                )
            if current_datatype in valid_datatypes:
                return func(*args, **kwargs)
            else:
                raise TypeError(
                    f'The check with name "{func.__name__}" only supports the following datatypes '
                    f'{[i.typeName() for i in acceptable_datatypes]} and not the given "{current_datatype()}" '
                    f"datatype"
                )

        return _wrapper

    return wrapper


def validate_scope(scope: ValidationScope):
    """This decorator decides if a function needs to be run or skipped based on params

    :param params: The configuration parameters to which define how pandera has to be used
    :param scope: the scope for which the function is valid. i.e. "DATA" scope function only works to validate the data,
                 "SCHEMA"  scope runs for schema checks function.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            def _get_check_obj():
                """
                Get dataframe object passed as arg to the decorated func.

                Returns:
                    The DataFrame object.
                """
                if args:
                    for value in args:
                        if isinstance(value, DataFrame):
                            return value

            config = get_config_context()
            if scope == ValidationScope.SCHEMA:
                if config.validation_depth in (
                    ValidationDepth.SCHEMA_AND_DATA,
                    ValidationDepth.SCHEMA_ONLY,
                ):
                    return func(self, *args, **kwargs)
                else:
                    warnings.warn(
                        f"Skipping execution of function {func.__name__} as validation depth is set to DATA_ONLY ",
                        stacklevel=2,
                    )
                    # If the function was skip, return the `check_obj` value anyway,
                    # given that some return value is expected
                    return _get_check_obj()

            elif scope == ValidationScope.DATA:
                if config.validation_depth in (
                    ValidationDepth.SCHEMA_AND_DATA,
                    ValidationDepth.DATA_ONLY,
                ):
                    return func(self, *args, **kwargs)
                else:
                    warnings.warn(
                        f"Skipping execution of function {func.__name__} as validation depth is set to SCHEMA_ONLY",
                        stacklevel=2,
                    )
                    # If the function was skip, return the `check_obj` value anyway,
                    # given that some return value is expected
                    return _get_check_obj()

        return wrapper

    return _wrapper


def cache_check_obj():
    """This decorator evaluates if `check_obj` should be cached before validation.

    As each new data check added to the Pandera schema by the user triggers a new
    Spark action, Spark reprocesses the `check_obj` DataFrame multiple times.
    To prevent this waste of processing resources and to reduce validation times in
    complex scenarios, the decorator created by this factory caches the `check_obj`
    DataFrame before validation and unpersists it afterwards.

    This decorator is meant to be used primarily in the `validate()` function
    entrypoint.

    The behavior of the resulting decorator depends on the `PANDERA_PYSPARK_CACHING` and
    `PANDERA_KEEP_CACHED_DATAFRAME` (optional) environment variables.

    Usage:
        @cache_check_obj()
        def validate(check_obj: DataFrame):
            # ...
    """

    def _wrapper(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Skip if not enabled
            if get_config_context().cache_dataframe is not True:
                return func(self, *args, **kwargs)

            check_obj: DataFrame = None

            # Check if decorated function has a dataframe object as an positional arg
            for arg in args:
                if isinstance(arg, DataFrame):
                    check_obj = arg
                    break

            # If it doesn't exist, fallback to kwargs and search for a `check_obj` key
            if check_obj is None:
                check_obj = kwargs.get("check_obj", None)

            if not isinstance(check_obj, DataFrame):
                raise ValueError(
                    "Expected to find a DataFrame object in a arg or a `check_obj` "
                    "kwarg in the decorated function "
                    f"`{func.__name__}`. Got {args=}/{kwargs=}"
                )

            @contextmanager
            def cached_check_obj():
                """Cache the dataframe and unpersist it after function execution."""
                logger.debug("Caching dataframe...")
                check_obj.cache()

                yield  # Execute the decorated function

                if not get_config_context().keep_cached_dataframe:
                    # If not cached, `.unpersist()` does nothing
                    logger.debug("Unpersisting dataframe...")
                    check_obj.unpersist()

            with cached_check_obj():
                return func(self, *args, **kwargs)

        return wrapper

    return _wrapper
