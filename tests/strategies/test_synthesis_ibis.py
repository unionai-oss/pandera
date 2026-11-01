"""Tests for Ibis data synthesis (native strategy() / example())."""

from __future__ import annotations

import pytest

import pandera.ibis as pa
from pandera.api.synthesis.registry import _reset_registry_for_tests
from pandera.strategies.faker_strategies import faker_backend


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    try:
        from pandera.api.synthesis import register_synthesis_backend

        register_synthesis_backend("faker", faker_backend)
        yield
    finally:
        _reset_registry_for_tests()


class _IbisModel(pa.DataFrameModel):
    name: str = pa.Field(
        semantic="name", str_length={"min_value": 3, "max_value": 20}
    )
    age: int = pa.Field(ge=18, le=99)


def test_default_example():
    schema = pa.DataFrameSchema({"x": pa.Column(int, checks=pa.Check.ge(0))})
    t = schema.example(size=10)
    assert t.count().execute() == 10


def test_realistic_opt_in_and_validates():
    t = _IbisModel.example(size=20, synthesis_backend="faker", seed=42)
    assert t.count().execute() == 20
    df = t.execute()
    assert df["age"].min() >= 18 and df["age"].max() <= 99
    assert _IbisModel.validate(t).count().execute() == 20


def test_seed_reproducible():
    a = _IbisModel.example(
        size=10, synthesis_backend="faker", seed=7
    ).execute()
    b = _IbisModel.example(
        size=10, synthesis_backend="faker", seed=7
    ).execute()
    assert a.equals(b)


def test_strategy_example():
    strat = _IbisModel.strategy(size=15, synthesis_backend="faker", seed=3)
    t = strat.example()
    assert t.count().execute() == 15
