# pylint: disable=wrong-import-position,wildcard-import,unused-wildcard-import
"""Unit tests for the deprecated top-level pandera DataFrameSchema class.

Delete this file once the top-level pandera._pandas_deprecated module is
removed.
"""

import pytest
from pandera._pandas_deprecated import DataFrameSchema as _DataFrameSchema


@pytest.fixture(autouse=True)
def monkeypatch_dataframe_schema(monkeypatch):
    """Monkeypatch DataFrameSchema before importing test_schemas"""
    monkeypatch.setattr(
        "tests.pandas.test_schemas.DataFrameSchema", _DataFrameSchema
    )


from tests.pandas.test_schemas import *
