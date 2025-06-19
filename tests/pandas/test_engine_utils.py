"""Unit tests for engine module utility functions."""

import numpy as np
import pandas as pd
import pytest

from pandera.engines import utils


@pytest.mark.parametrize(
    "data_container, data_type, expected_failure_cases",
    [
        [pd.Series(list("ab1cd3")), int, [False, False, True] * 2],
        [pd.Series(list("12345")), int, [True] * 5],
        [pd.Series([1, 2, "foo", "bar"]), float, [True, True, False, False]],
    ],
)
def test_numpy_pandas_coercible(
    data_container, data_type, expected_failure_cases
):
    """Test that the correct boolean Series outputs are returned."""
    assert (
        expected_failure_cases
        == utils.numpy_pandas_coercible(data_container, data_type).tolist()
    )


@pytest.mark.parametrize(
    "data_container",
    [
        pd.Series([1, 2, 3, 4]),
        np.array([1, 2, 3, 4]),
        pd.DataFrame({0: [1, 2, 3, 4]}),
        np.array([[1], [2], [3], [4]]),
    ],
)
def test_numpy_pandas_coerce_failure_cases(data_container):
    """
    Test that different data container types can be checked for coerce failure
    cases.
    """
    failure_cases = utils.numpy_pandas_coerce_failure_cases(
        data_container, int
    )
    assert failure_cases is None


@pytest.mark.parametrize(
    "invalid_data_container, exception_type",
    [
        [1, TypeError],
        [5.1, TypeError],
        ["foobar", TypeError],
        [[1, 2, 3], TypeError],
        [{0: 1}, TypeError],
        # pylint: disable=too-many-function-args
        [np.array([1]).reshape(1, 1, 1), ValueError],
    ],
)
def test_numpy_pandas_coerce_failure_cases_exceptions(
    invalid_data_container, exception_type
):
    """
    Test exceptions of trying to get failure cases for invalid input types.
    """
    error_msg = {
        TypeError: "type of data_container .+ not understood",
        ValueError: "only numpy arrays of 1 or 2 dimensions are supported",
    }[exception_type]
    with pytest.raises(exception_type, match=error_msg):
        utils.numpy_pandas_coerce_failure_cases(invalid_data_container, int)
