"""Built-in checks for polars."""

from typing import Any, Tuple

import polars as pl

from pandera.api.extensions import register_builtin_check
from pandera.api.polars.types import PolarsData
from pandera.backends.polars.constants import CHECK_OUTPUT_KEY


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(data: PolarsData, min_value: Any) -> pl.LazyFrame:
    """Ensure all elements of a data container equal a certain value.

    :param value: values in this pandas data structure must be
        equal to this value.
    """
    return data.dataframe.with_columns(
        [pl.col(data.key).ge(min_value).alias(CHECK_OUTPUT_KEY)]
    )
