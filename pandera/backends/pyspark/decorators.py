"""This module holds the decorators only valid for pyspark"""

import functools
import warnings
from enum import Enum
from typing import List, Type

import pyspark.sql

from pandera.api.pyspark.types import PysparkDefaultTypes
from pandera.config import CONFIG, ValidationDepth
from pandera.errors import SchemaError


class ValidationScope(Enum):
    """Indicates whether a check/validator operates at a schema of data level."""

    SCHEMA = "schema"
    DATA = "data"


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
                    f" This error is usually caused by schema mismatch the value is different from schema defined in"
                    f" pandera schema and one in the dataframe",
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
            if scope == ValidationScope.SCHEMA:
                if CONFIG.validation_depth in (
                    ValidationDepth.SCHEMA_AND_DATA,
                    ValidationDepth.SCHEMA_ONLY,
                ):
                    return func(self, *args, **kwargs)
                else:
                    warnings.warn(
                        "Skipping Execution of function as parameters set to DATA_ONLY ",
                        stacklevel=2,
                    )
                    if not kwargs:
                        for value in kwargs.values():
                            if isinstance(value, pyspark.sql.DataFrame):
                                return value
                    if args:
                        for value in args:
                            if isinstance(value, pyspark.sql.DataFrame):
                                return value

            elif scope == ValidationScope.DATA:
                if CONFIG.validation_depth in (
                    ValidationDepth.SCHEMA_AND_DATA,
                    ValidationDepth.DATA_ONLY,
                ):
                    return func(self, *args, **kwargs)
                else:
                    warnings.warn(
                        "Skipping Execution of function as parameters set to SCHEMA_ONLY ",
                        stacklevel=2,
                    )

        return wrapper

    return _wrapper
