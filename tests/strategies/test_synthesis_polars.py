"""Tests for Polars data synthesis (native strategy() / example())."""

from __future__ import annotations

import re

import pytest

import pandera.polars as pa
from pandera.api.synthesis.registry import _reset_registry_for_tests
from pandera.strategies.faker_strategies import faker_backend

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    try:
        from pandera.api.synthesis import register_synthesis_backend

        register_synthesis_backend("faker", faker_backend)
        yield
    finally:
        _reset_registry_for_tests()


class _PolarsModel(pa.DataFrameModel):
    name: str = pa.Field(
        semantic="name", str_length={"min_value": 3, "max_value": 30}
    )
    email: str = pa.Field(semantic="email", str_matches=EMAIL_RE)
    age: int = pa.Field(ge=18, le=99)


def test_default_example():
    schema = pa.DataFrameSchema({"x": pa.Column(int, checks=pa.Check.ge(0))})
    df = schema.example(size=10)
    assert df.shape == (10, 1)
    assert schema.validate(df).shape == df.shape


def test_realistic_opt_in_and_validates():
    df = _PolarsModel.example(size=20, synthesis_backend="faker", seed=42)
    assert df.shape[0] == 20
    assert all(re.fullmatch(EMAIL_RE, e) for e in df["email"])
    assert df["age"].min() >= 18 and df["age"].max() <= 99
    assert _PolarsModel.validate(df).shape == df.shape


def test_seed_reproducible():
    a = _PolarsModel.example(size=10, synthesis_backend="faker", seed=7)
    b = _PolarsModel.example(size=10, synthesis_backend="faker", seed=7)
    assert a.equals(b)


def test_strategy_example():
    strat = _PolarsModel.strategy(size=15, synthesis_backend="faker", seed=3)
    df = strat.example()
    assert df.shape[0] == 15
    assert _PolarsModel.validate(df).shape == df.shape


def test_default_constraint_bounds():
    # Without a backend, built-in bounds still hold via the shared core.
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, checks=pa.Check.in_range(1, 100))}
    )
    df = schema.example(size=30, seed=1)
    assert df["a"].min() >= 1 and df["a"].max() <= 100
