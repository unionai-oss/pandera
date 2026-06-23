"""Unit tests for ``pandera.backends.register_checks``.

These cover the import-guard fallback added in GH #2387 without needing a
numpy/pandas-free environment, so they run (and report coverage) in the
default pandas test matrix. The polars-only regression lives in
``tests/polars/test_polars_no_pandas.py``.
"""

import builtins

import pytest

from pandera.backends import register_checks


@pytest.fixture
def clear_loader_cache():
    """Reset the ``lru_cache`` before and after so the import is re-attempted."""
    register_checks._load_get_backend_types_from_mro.cache_clear()
    try:
        yield
    finally:
        register_checks._load_get_backend_types_from_mro.cache_clear()


@pytest.mark.usefixtures("clear_loader_cache")
def test_loader_returns_callable_when_pandas_importable():
    """With pandas present the loader returns the real callable."""
    loader = register_checks._load_get_backend_types_from_mro()
    assert callable(loader)


@pytest.mark.usefixtures("clear_loader_cache")
def test_loader_returns_none_on_import_error(monkeypatch):
    """An ImportError importing the pandas types makes the loader return None.

    Simulates the ``pandera[polars]``-only install where
    ``pandera.api.pandas.types`` cannot be imported (GH #2387).
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandera.api.pandas.types":
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert register_checks._load_get_backend_types_from_mro() is None


def test_pyspark_pandas_does_not_route_to_pyspark_sql(monkeypatch):
    """``pyspark.pandas`` frames must not hit the ``pyspark.sql`` dispatch.

    With pandas absent the pandas-MRO check returns None, so dispatch falls
    through to the native prefix checks. The pyspark branch matches
    ``pyspark.sql`` (not bare ``pyspark``) so a ``pyspark.pandas`` frame falls
    through instead of misrouting to the pyspark-sql backend. See #2387.
    """
    monkeypatch.setattr(
        register_checks,
        "_load_get_backend_types_from_mro",
        lambda: None,
    )

    called = []
    monkeypatch.setattr(
        "pandera.backends.pyspark.register.register_pyspark_backends",
        lambda *a, **k: called.append(True),
    )

    class FakePysparkPandasFrame:
        pass

    FakePysparkPandasFrame.__module__ = "pyspark.pandas.frame"

    register_checks.register_default_check_backends(FakePysparkPandasFrame)
    assert called == []
