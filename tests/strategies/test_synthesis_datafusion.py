"""Tests for DataFusion data synthesis (native strategy() / example())."""

from __future__ import annotations

import re

import pytest

import pandera.datafusion as pa
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


class _DataFusionModel(pa.DataFrameModel):
    name: str = pa.Field(
        semantic="name", str_length={"min_value": 3, "max_value": 30}
    )
    email: str = pa.Field(semantic="email", str_matches=EMAIL_RE)
    age: int = pa.Field(ge=18, le=99)


def _to_dict(df):
    """Materialize a datafusion.DataFrame as a ``{col: [values]}`` dict."""
    return df.to_arrow_table().to_pydict()


def test_default_example():
    schema = pa.DataFrameSchema({"x": pa.Column(int, checks=pa.Check.ge(0))})
    df = schema.example(size=10)
    assert df.to_arrow_table().num_rows == 10
    assert schema.validate(df).to_arrow_table().num_rows == 10


def test_realistic_opt_in_and_validates():
    df = _DataFusionModel.example(size=20, synthesis_backend="faker", seed=42)
    data = _to_dict(df)
    assert len(data["age"]) == 20
    assert all(re.fullmatch(EMAIL_RE, e) for e in data["email"])
    assert min(data["age"]) >= 18 and max(data["age"]) <= 99
    assert all(3 <= len(n) <= 30 for n in data["name"])
    assert _DataFusionModel.validate(df).to_arrow_table().num_rows == 20


def test_seed_reproducible():
    a = _DataFusionModel.example(size=10, synthesis_backend="faker", seed=7)
    b = _DataFusionModel.example(size=10, synthesis_backend="faker", seed=7)
    assert a.to_arrow_table().equals(b.to_arrow_table())


def test_strategy_example():
    strat = _DataFusionModel.strategy(
        size=15, synthesis_backend="faker", seed=3
    )
    df = strat.example()
    assert df.to_arrow_table().num_rows == 15


def test_default_constraint_bounds():
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, checks=pa.Check.in_range(1, 100))}
    )
    df = schema.example(size=30, seed=1)
    data = schema.validate(df).to_arrow_table().to_pydict()
    assert min(data["a"]) >= 1 and max(data["a"]) <= 100
