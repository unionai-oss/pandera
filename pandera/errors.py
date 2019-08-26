"""pandera-specific errors."""


class SchemaInitError(Exception):
    pass


class SchemaDefinitionError(Exception):
    pass


class SchemaError(Exception):
    pass
