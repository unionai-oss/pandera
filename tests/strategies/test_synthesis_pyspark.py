"""Tests for PySpark data synthesis (native strategy() / example())."""

from __future__ import annotations

from typing import Any

import pytest

pa: Any = pytest.importorskip("pandera.pyspark")

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


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    yield SparkSession.getActiveSession() or (
        SparkSession.builder.master("local[1]")
        .appName("pandera-data-synthesis-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


class _PysparkModel(pa.DataFrameModel):
    name: str = pa.Field(
        semantic="name", str_length={"min_value": 3, "max_value": 20}
    )
    age: int = pa.Field(ge=18, le=99)


def test_default_example(spark):
    schema = pa.DataFrameSchema({"x": pa.Column(int, checks=pa.Check.ge(0))})
    sdf = schema.example(size=10)
    assert sdf.count() == 10


def test_realistic_opt_in_and_validates(spark):
    sdf = _PysparkModel.example(size=20, synthesis_backend="faker", seed=42)
    assert sdf.count() == 20
    pdf = sdf.toPandas()
    assert pdf["age"].min() >= 18 and pdf["age"].max() <= 99
    assert _PysparkModel.validate(sdf).count() == 20


def test_seed_reproducible(spark):
    a = _PysparkModel.example(
        size=10, synthesis_backend="faker", seed=7
    ).collect()
    b = _PysparkModel.example(
        size=10, synthesis_backend="faker", seed=7
    ).collect()
    assert a == b
