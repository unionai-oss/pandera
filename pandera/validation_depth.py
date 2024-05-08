"""Map reason_code to ValidationScope depth type"""

import functools
import logging

from pandera.backends.base import CoreCheckResult
from pandera.config import ValidationDepth, ValidationScope, get_config_context
from pandera.errors import SchemaErrorReason

logger = logging.getLogger(__name__)


VALIDATION_DEPTH_ERROR_CODE_MAP = {
    SchemaErrorReason.INVALID_TYPE: ValidationScope.DATA,
    SchemaErrorReason.DATATYPE_COERCION: ValidationScope.DATA,
    SchemaErrorReason.COLUMN_NOT_IN_SCHEMA: ValidationScope.SCHEMA,
    SchemaErrorReason.COLUMN_NOT_ORDERED: ValidationScope.SCHEMA,
    SchemaErrorReason.DUPLICATE_COLUMN_LABELS: ValidationScope.SCHEMA,
    SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME: ValidationScope.SCHEMA,
    SchemaErrorReason.SCHEMA_COMPONENT_CHECK: ValidationScope.SCHEMA,
    SchemaErrorReason.DATAFRAME_CHECK: ValidationScope.DATA,
    SchemaErrorReason.CHECK_ERROR: ValidationScope.DATA,
    SchemaErrorReason.DUPLICATES: ValidationScope.DATA,
    SchemaErrorReason.WRONG_FIELD_NAME: ValidationScope.SCHEMA,
    SchemaErrorReason.SERIES_CONTAINS_NULLS: ValidationScope.SCHEMA,
    SchemaErrorReason.SERIES_CONTAINS_DUPLICATES: ValidationScope.DATA,
    SchemaErrorReason.WRONG_DATATYPE: ValidationScope.SCHEMA,
    SchemaErrorReason.NO_ERROR: ValidationScope.SCHEMA,
    SchemaErrorReason.ADD_MISSING_COLUMN_NO_DEFAULT: ValidationScope.DATA,
    SchemaErrorReason.INVALID_COLUMN_NAME: ValidationScope.SCHEMA,
    SchemaErrorReason.MISMATCH_INDEX: ValidationScope.DATA,
    SchemaErrorReason.PARSER_ERROR: ValidationScope.DATA,
}


def validation_type(schema_error_reason):
    """Map a reason_code to a ValidationScope depth type

    :param SchemaErrorReason: schema error reason enum
    :returns ValidationScope: validation depth enum
    """
    return VALIDATION_DEPTH_ERROR_CODE_MAP[schema_error_reason]


def validate_scope(scope: ValidationScope):
    """This decorator decides if a function needs to be run or skipped based on params

    :param params: The configuration parameters to which define how pandera has
        to be used
    :param scope: the scope for which the function is valid. i.e. "DATA" scope
        function only works to validate the data values, "SCHEMA"  scope runs for
        checks at the schema/metadata level.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def wrapper(self, check_obj, *args, **kwargs):

            config = get_config_context()

            if scope == ValidationScope.SCHEMA:
                if config.validation_depth == ValidationDepth.DATA_ONLY:
                    logger.debug(
                        f"Skipping execution of check {func.__name__} since "
                        "validation depth is set to DATA_ONLY.",
                        stacklevel=2,
                    )
                    return CoreCheckResult(passed=True)
                return func(self, check_obj, *args, **kwargs)

            elif scope == ValidationScope.DATA:
                if config.validation_depth == ValidationDepth.SCHEMA_ONLY:
                    logger.debug(
                        f"Skipping execution of check {func.__name__} since "
                        "validation depth is set to SCHEMA_ONLY",
                        stacklevel=2,
                    )
                    return CoreCheckResult(passed=True)
                return func(self, check_obj, *args, **kwargs)

        return wrapper

    return _wrapper
