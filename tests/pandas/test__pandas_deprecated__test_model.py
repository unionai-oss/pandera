# pylint: disable=wrong-import-position,wildcard-import,unused-wildcard-import
"""Unit tests for the deprecated top-level pandera DataFrameModel class.

Delete this file once the top-level pandera._pandas_deprecated module is
removed.
"""

import pytest
from pandera._pandas_deprecated import DataFrameModel as _DataFrameModel


@pytest.fixture(autouse=True)
def monkeypatch_dataframe_model(monkeypatch):
    """Monkeypatch DataFrameModel before importing test_schemas"""
    monkeypatch.setattr(
        "tests.pandas.test_schemas.DataFrameModel", _DataFrameModel
    )


from tests.pandas.test_schemas import *
