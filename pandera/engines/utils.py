"""Engine module utilities."""

import itertools
from typing import Any, Union

import numpy as np
import pandas as pd

from .. import check_utils
from .type_aliases import PandasObject


def numpy_pandas_coercible(series: pd.Series, type_: Any) -> pd.Series:
    """Checks whether a series is coercible with respect to a type.

    Bisects the series until all the failure cases are found.

    NOTE: this does not account for koalas .astype behavior, which defaults
    to converting uncastable values to NA values.
    """

    def _bisect(series):
        assert (
            series.shape[0] >= 2
        ), "cannot bisect a pandas Series of length < 2"
        bisect_index = series.shape[0] // 2
        return [series.iloc[:bisect_index], series.iloc[bisect_index:]]

    def _coercible(series):
        try:
            series.astype(type_)
            return True
        except (ValueError, TypeError):
            return False

    search_list = [series] if series.size == 1 else _bisect(series)
    failure_index = []
    while search_list:
        candidates = []
        for _series in search_list:
            if _series.shape[0] == 1 and not _coercible(_series):
                # if series is reduced to a single value and isn't coercible,
                # keep track of its index value.
                failure_index.append(_series.index.item())
            elif not _coercible(_series):
                # if the series length > 1, add it to the candidates list
                # to be further bisected
                candidates.append(_series)

        # the new search list is a flat list of bisected series views.
        search_list = list(
            itertools.chain.from_iterable([_bisect(c) for c in candidates])
        )

    # NOTE: this is a hack to support koalas
    if type(series).__module__.startswith("databricks.koalas"):
        out = type(series)(
            series.index.isin(failure_index).to_series().to_numpy(),  # type: ignore[union-attr]
            index=series.index.values.to_numpy(),
            name=series.name,
        )
        out.index.name = series.index.name
        return out
    return pd.Series(~series.index.isin(failure_index), index=series.index)


def numpy_pandas_coerce_failure_cases(
    data_container: Union[PandasObject, np.ndarray], type_: Any
) -> PandasObject:
    """
    Get the failure cases resulting from trying to coerce a pandas/numpy object
    into particular data type.
    """
    # pylint: disable=import-outside-toplevel,cyclic-import
    from pandera import error_formatters

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
            args=(type_,),
        )
        _, failure_cases = check_utils.prepare_dataframe_check_output(
            data_container,
            check_output,
            ignore_na=False,
        )
    elif check_utils.is_field(data_container):
        check_output = numpy_pandas_coercible(data_container, type_)
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
