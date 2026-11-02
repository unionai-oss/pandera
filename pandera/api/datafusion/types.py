"""DataFusion types."""

from typing import NamedTuple, Union

import datafusion
import pyarrow as pa


class DataFusionData(NamedTuple):
    """Data container passed to ``native=True`` DataFusion checks.

    Mirrors :class:`~pandera.api.polars.types.PolarsData` and
    :class:`~pandera.api.pyarrow.types.PyArrowData` so that check functions
    taking a single positional argument receive the same shape across
    backends. ``dataframe`` is the lazy ``datafusion.DataFrame`` being
    validated and ``key`` the column under check (``"*"`` for
    dataframe-level checks).
    """

    dataframe: datafusion.DataFrame
    key: str = "*"


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: datafusion.DataFrame
    check_passed: datafusion.DataFrame
    checked_object: datafusion.DataFrame
    failure_cases: datafusion.DataFrame


DataFusionCheckObjects = datafusion.DataFrame

DataFusionDtypeInputTypes = Union[
    str,
    type,
    pa.DataType,
]
