"""Registers fixtures for core"""

import os
from typing import Generator

import pytest
from pandera.api.checks import Check

# pylint: disable=unused-import
ENGINES = os.getenv("CI_MODIN_ENGINES", "").split(",")
if ENGINES == [""]:
    ENGINES = ["dask"]


@pytest.fixture(scope="function")
def custom_check_teardown() -> Generator[None, None, None]:
    """Remove all custom checks after execution of each pytest function."""
    yield
    for check_name in list(Check.REGISTERED_CUSTOM_CHECKS):
        del Check.REGISTERED_CUSTOM_CHECKS[check_name]


@pytest.fixture(scope="session", params=ENGINES, autouse=True)
def setup_modin_engine(request):
    """Set up the modin engine.

    Eventually this will also support dask execution backend.
    """
    engine = request.param
    os.environ["MODIN_ENGINE"] = engine
    os.environ["MODIN_STORAGE_FORMAT"] = "pandas"
    os.environ["MODIN_MEMORY"] = "100000000"
    os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

    if engine == "ray":
        # pylint: disable=import-outside-toplevel
        import ray

        ray.init(
            runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}}
        )
        yield
        ray.shutdown()

    elif engine == "dask":
        # pylint: disable=import-outside-toplevel
        from distributed import Client

        client = Client()
        yield
        client.close()
    else:
        raise ValueError(f"Not supported engine: {engine}")
