"""Unit tests of cudf accessor functionality.
"""

import pytest

from pandera import cudf_accessor


# pylint: disable=too-few-public-methods
class CustomAccessor:
    """Mock accessor class"""

    def __init__(self, obj):
        self._obj = obj


def test_cudf_accessor_warning():
    """Test that cudf accessor raises warning when name already exists."""
    cudf_accessor.register_dataframe_accessor("foo")(CustomAccessor)
    with pytest.warns(UserWarning):
        cudf_accessor.register_dataframe_accessor("foo")(CustomAccessor)
