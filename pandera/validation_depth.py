"""Map reason_code to ValidationScope depth type"""

from enum import Enum

from pandera.errors import SchemaErrorReason


class ValidationScope(Enum):
    """Indicates whether a check/validator operates at a schema of data level."""

    SCHEMA = "schema"
    DATA = "data"


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
}


def validation_type(schema_error_reason):
    """Map a reason_code to a ValidationScope depth type

    :param SchemaErrorReason: schema error reason enum
    :returns ValidationScope: validation depth enum
    """
    return VALIDATION_DEPTH_ERROR_CODE_MAP[schema_error_reason]
