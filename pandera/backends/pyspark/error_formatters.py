"""Make schema error messages human-friendly."""


def format_generic_error_message(
    parent_schema,
    check,
) -> str:
    """Construct an error message when a check validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    """
    return f"{parent_schema} failed validation " f"{check.error}"


def scalar_failure_case(x) -> dict:
    """Construct failure case from a scalar value.

    :param x: a scalar value representing failure case.
    :returns: Dictionary used for error reporting with ``SchemaErrors``.
    """
    return {
        "index": [None],
        "failure_case": [x],
    }
