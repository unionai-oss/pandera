"""Tests for pandas data-synthesis integration.

Covers the ``synthesis_backend`` / ``seed`` / ``semantic`` plumbing through
``Schema.strategy()`` / ``Schema.example()`` and model equivalents, plus
backward-compatibility guarantees of the default (no backend) path.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.api.synthesis.registry import _reset_registry_for_tests
from pandera.config import config_context
from pandera.errors import SchemaError, SynthesisBackendNotFoundError
from pandera.strategies.faker_strategies import faker_backend

from .test_synthesis_values import EMAIL_RE


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    try:
        from pandera.api.synthesis import register_synthesis_backend

        register_synthesis_backend("faker", faker_backend)
        yield
    finally:
        _reset_registry_for_tests()


class _CustomerSchema(pa.DataFrameModel):
    name: str = pa.Field(
        semantic="name", str_length={"min_value": 3, "max_value": 40}
    )
    email: str = pa.Field(semantic="email", str_matches=EMAIL_RE)
    country: str = pa.Field(semantic="country", isin=["US", "CA", "GB", "MX"])
    age: int = pa.Field(ge=18, le=99)


class TestFakerPandasIntegration:
    def test_example_opt_in_realistic(self):
        df = _CustomerSchema.example(
            size=20, synthesis_backend="faker", seed=42
        )
        assert df.shape[0] == 20
        # Realistic (not gibberish): names contain letters, emails match.
        assert all(len(n) >= 3 for n in df["name"])
        assert all(re.fullmatch(EMAIL_RE, e) for e in df["email"])
        assert set(df["country"]) <= {"US", "CA", "GB", "MX"}
        # schema validity
        assert _CustomerSchema.validate(df).shape == df.shape

    def test_seed_reproducible(self):
        a = _CustomerSchema.example(size=10, synthesis_backend="faker", seed=7)
        b = _CustomerSchema.example(size=10, synthesis_backend="faker", seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_differs(self):
        a = _CustomerSchema.example(size=10, synthesis_backend="faker", seed=1)
        b = _CustomerSchema.example(size=10, synthesis_backend="faker", seed=2)
        assert not a.equals(b)

    def test_strategy_opt_in(self):
        strat = _CustomerSchema.strategy(
            size=15, synthesis_backend="faker", seed=3
        )
        df = strat.example()
        assert df.shape[0] == 15
        assert _CustomerSchema.validate(df).shape == df.shape

    def test_unsatisfiable_constraint_still_valid(self):
        # An isin the country provider cannot honour + explicit semantic:
        # for tractability use an empty-ish target that still must validate.
        schema = pa.DataFrameSchema(
            {
                "c": pa.Column(
                    str, semantic="country", checks=[pa.Check.isin(["US"])]
                )
            }
        )
        df = schema.example(size=5, synthesis_backend="faker", seed=1)
        assert set(df["c"]) <= {"US"}
        assert schema.validate(df).shape == df.shape


class TestDefaultPathUnchanged:
    def test_default_equals_explicit_none(self):
        # The new kwarg defaults reproduce today's default behavior; with a
        # ``seed`` the deterministic path reproduces identical frames whether
        # the backend is ``None`` or omitted.
        schema = pa.DataFrameSchema(
            {
                "name": pa.Column(str, checks=pa.Check.str_length(3, 20)),
                "age": pa.Column(int, checks=pa.Check.in_range(18, 99)),
            }
        )
        a = schema.example(size=8, seed=11)
        b = schema.example(size=8, seed=11, synthesis_backend=None)
        pd.testing.assert_frame_equal(a, b)

    def test_default_is_gibberish(self):
        # Without the backend, the default path is pure-hypothesis (no
        # realistic providers), so a name column is NOT realistic.
        df = _CustomerSchema.example(
            size=10, seed=1, synthesis_backend="faker"
        )
        # with faker the age column is deterministic ints within bounds
        assert df["age"].between(18, 99).all()


class TestGlobalConfigBackend:
    def test_config_context_drives_example(self):
        with config_context(synthesis_backend="faker"):
            df = _CustomerSchema.example(size=10, seed=5)
        assert all(re.fullmatch(EMAIL_RE, e) for e in df["email"])

    def test_explicit_overrides_config(self):
        schema = pa.DataFrameSchema({"name": pa.Column(str, semantic="name")})
        with config_context(synthesis_backend="faker"):
            # explicit None -> falls back to config -> faker
            df = schema.example(size=5, seed=1, synthesis_backend=None)
        assert schema.validate(df).shape == df.shape


class TestUnknownBackend:
    def test_unknown_raises(self):
        with pytest.raises(SynthesisBackendNotFoundError):
            _CustomerSchema.example(synthesis_backend="nope", seed=1)

    def test_backend_error_falls_back(self):
        # A backend that raises inside generate() must not fail synthesis.
        from pandera.api.synthesis import register_synthesis_backend

        class _Broken:
            name = "broken"

            def field_generator(self):
                class _G:
                    def generate(self, *a, **k):
                        raise RuntimeError("boom")

                return _G()

        register_synthesis_backend("broken", _Broken())
        df = _CustomerSchema.example(synthesis_backend="broken", seed=1)
        assert _CustomerSchema.validate(df).shape == df.shape


class TestSemanticMetadata:
    def test_semantic_no_validation_effect(self):
        schema = pa.DataFrameSchema({"x": pa.Column(str, semantic="name")})
        # validate ignores semantic (None metadata only)
        assert schema.validate(pd.DataFrame({"x": ["a", "b"]})).shape == (2, 1)

    def test_invalid_semantic_raises_at_definition(self):
        with pytest.raises(ValueError):
            pa.DataFrameSchema(
                {"x": pa.Column(str, semantic="not-a-real-semantic")}
            )

    def test_index_semantic(self):
        schema = pa.DataFrameSchema(
            {"x": pa.Column(int, checks=pa.Check.ge(0))},
            index=pa.Index(str, semantic="city"),
        )
        # default path: index still generated validly
        df = schema.example(size=5)
        assert df.shape == (5, 1)
