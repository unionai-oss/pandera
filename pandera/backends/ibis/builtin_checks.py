"""Built-in checks for Ibis."""

from typing import Any, TypeVar

import ibis
import ibis.expr.types as ir

from pandera.api.extensions import register_builtin_check
from pandera.api.ibis.types import IbisData

T = TypeVar("T")


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: IbisData, value: Any) -> ir.Table:
    """Ensure all elements of a data container equal a certain value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "key".
    :param value: values in this polars data structure must be
        equal to this value.
    """
    return data.table[data.key] == value
