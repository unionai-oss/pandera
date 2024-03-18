import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["PANDERA_VALIDATION_DEPTH"] = "SCHEMA_AND_DATA"
