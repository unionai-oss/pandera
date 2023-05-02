"""Make schema error messages human-friendly."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pandera.errors import SchemaErrorReason

# import pandas as pd


def format_generic_error_message(
    parent_schema,
    check,
) -> str:
    """Construct an error message when a check validator fails.

    :param parent_schema: class of schema being validated.
    :param check: check that generated error.
    :param check_index: The validator that failed.
    """
    return f"{parent_schema} failed validation " f"{check}"



def scalar_failure_case(x) -> dict:
    """Construct failure case from a scalar value.

    :param x: a scalar value representing failure case.
    :returns: DataFrame used for error reporting with ``SchemaErrors``.
    """
    return {
        "index": [None],
        "failure_case": [x],
    }

