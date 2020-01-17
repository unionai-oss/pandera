"""pandera-specific errors."""


class SchemaInitError(Exception):
    """Raised when schema initialization fails."""


class SchemaDefinitionError(Exception):
    """Raised when schema definition is invalid on object validation."""


class SchemaError(Exception):
    """Raised when object does not pass schema validation constraints."""
