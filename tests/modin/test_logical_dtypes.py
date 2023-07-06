"""Tests logical dtypes."""

from types import ModuleType
from typing import Generator

import modin.pandas as mpd

# pylint: disable=wildcard-import, unused-wildcard-import
from tests.core.test_logical_dtypes import *


@pytest.fixture(scope="module")  # type: ignore
def datacontainer_lib() -> (
    Generator[ModuleType, None, None]
):  # pylint: disable=function-redefined
    """Yield the modin.pandas module"""
    yield mpd
