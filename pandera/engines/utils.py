"""Engine module utilities."""

from typing import Any, Union

import numpy as np
import pandas as pd

from .. import check_utils
from .type_aliases import PandasObject


def numpy_pandas_coercible(series: pd.Series, type_: Any) -> pd.Series:
    """Checks whether a series is coercible with respect to a type.

    Bisects the series until all the failure cases are found.

    NOTE: this does not account for pyspark.pandas .astype behavior, which
    defaults to converting uncastable values to NA values.
    """
    # pylint: disable=import-outside-toplevel,cyclic-import
    from pandera.engines import pandas_engine

    data_type = pandas_engine.Engine.dtype(type_)

    def _coercible(x):
        try:
            data_type.coerce_value(x)
            return True
        except Exception:  # pylint:disable=broad-except
            return False

    return series.map(_coercible)


def numpy_pandas_coerce_failure_cases(
    data_container: Union[PandasObject, np.ndarray], type_: Any
) -> PandasObject:
    """
    Get the failure cases resulting from trying to coerce a pandas/numpy object
    into particular data type.
    """
    # pylint: disable=import-outside-toplevel,cyclic-import
    from pandera import error_formatters
    from pandera.engines import pandas_engine

    data_type = pandas_engine.Engine.dtype(type_)

    if isinstance(data_container, np.ndarray):
        if len(data_container.shape) == 1:
            data_container = pd.Series(data_container)
        elif len(data_container.shape) == 2:
            data_container = pd.DataFrame(data_container)
        else:
            raise ValueError(
                "only numpy arrays of 1 or 2 dimensions are supported"
            )

    if check_utils.is_index(data_container):
        data_container = data_container.to_series()  # type: ignore[union-attr]

    if check_utils.is_table(data_container):
        check_output = data_container.apply(  # type: ignore[union-attr]
            numpy_pandas_coercible,
            args=(data_type,),
        )
        _, failure_cases = check_utils.prepare_dataframe_check_output(
            data_container,
            check_output,
            ignore_na=False,
        )
    elif check_utils.is_field(data_container):
        check_output = numpy_pandas_coercible(data_container, data_type)
        _, failure_cases = check_utils.prepare_series_check_output(
            data_container,
            check_output,
            ignore_na=False,
        )
    else:
        raise TypeError(
            f"type of data_container {type(data_container)} not understood. "
            "Must be a pandas Series, Index, or DataFrame."
        )
    return error_formatters.reshape_failure_cases(
        failure_cases, ignore_na=False
    )
