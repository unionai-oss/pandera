"""Unit tests of modin accessor functionality.

Since modin doesn't currently support the pandas accessor extension API,
pandera implements it.
"""

import pytest

from pandera.accessors import modin_accessor


# pylint: disable=too-few-public-methods
class CustomAccessor:
    """Mock accessor class"""

    def __init__(self, obj):
        self._obj = obj


def test_modin_accessor_warning():
    """Test that modin accessor raises warning when name already exists."""
    modin_accessor.register_dataframe_accessor("foo")(CustomAccessor)
    with pytest.warns(UserWarning):
        modin_accessor.register_dataframe_accessor("foo")(CustomAccessor)
