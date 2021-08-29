"""Pytest fixtures for testing custom checks."""
from typing import Generator
from unittest import mock

import pandas as pd
import pytest

import pandera as pa
import pandera.extensions as pa_ext

__all__ = "custom_check_teardown", "extra_registered_checks"


@pytest.fixture(scope="function")
def custom_check_teardown() -> Generator[None, None, None]:
    """Remove all custom checks after execution of each pytest function."""
    yield
    for check_name in list(pa.Check.REGISTERED_CUSTOM_CHECKS):
        del pa.Check.REGISTERED_CUSTOM_CHECKS[check_name]


@pytest.fixture(scope="function")
def extra_registered_checks() -> Generator[None, None, None]:
    """temporarily registers custom checks onto the Check class"""
    # pylint: disable=unused-variable
    with mock.patch(
        "pandera.Check.REGISTERED_CUSTOM_CHECKS", new_callable=dict
    ):
        # register custom checks here
        @pa_ext.register_check_method()
        def no_param_check(_: pd.DataFrame) -> bool:
            return True

        yield
