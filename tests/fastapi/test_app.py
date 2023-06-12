# pylint: disable=redefined-outer-name,unused-argument
"""Unit tests for using pandera types in fastapi endpoints."""

import io
import subprocess
import time
from copy import deepcopy

import pandas as pd
import pytest
import requests
from hypothesis import given

from tests.fastapi.models import Transactions, TransactionsOut


@pytest.fixture(scope="module")
def app():
    """Transient app server for testing."""
    # pylint: disable=consider-using-with
    process = subprocess.Popen(
        ["uvicorn", "tests.fastapi.app:app", "--port", "8000"],
        stdout=subprocess.PIPE,
    )
    _wait_to_exist()
    yield process
    process.terminate()


def _wait_to_exist():
    for _ in range(20):
        try:
            requests.post("http://127.0.0.1:8000/")
            break
        except Exception:  # pylint: disable=broad-except
            time.sleep(3.0)


def test_items_endpoint(app):
    """Happy path test with pydantic type annotations."""
    data = {"name": "Book", "value": 10, "description": "Hello"}
    for _ in range(10):
        response = requests.post("http://127.0.0.1:8000/items/", json=data)
        if response.status_code != 200:
            time.sleep(3.0)
    assert response.json() == data


def test_transactions_endpoint(app):
    """Happy path test with pandera type endpoint type annotation."""
    data = {"id": [1], "cost": [10.99]}
    response = requests.post(
        "http://127.0.0.1:8000/transactions/",
        json=data,
    )
    expected_output = deepcopy(data)
    expected_output = [{"id": 1, "cost": 10.99, "name": "foo"}]
    assert response.json() == expected_output


@given(Transactions.strategy(size=10))
def test_upload_file_endpoint(app, sample):
    """
    Test upload file endpoint with Upload[DataFrame[DataFrameModel]] input.
    """
    buf = io.BytesIO()
    sample.to_parquet(buf)
    buf.seek(0)

    expected_result = pd.read_parquet(buf).assign(name="foo")
    buf.seek(0)

    response = requests.post(
        "http://127.0.0.1:8000/file/", files={"file": buf}
    )
    output = response.json()
    assert output["filename"] == "file"
    output_df = pd.read_json(output["df"])
    cost_notna = ~output_df["cost"].isna()
    pd.testing.assert_frame_equal(
        TransactionsOut.validate(output_df[cost_notna]),
        TransactionsOut.validate(expected_result[cost_notna]),
    )
