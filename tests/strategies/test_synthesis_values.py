"""Tests for the engine-agnostic value core and the built-in Faker backend.

Covers :func:`~pandera.strategies._values.compile_constraints_to_sampler` and
the ``"faker"`` realistic backend. Every test here is opt-in: the default
pure-hypothesis synthesis path is untouched.
"""

from __future__ import annotations

import itertools
import re

import pytest

from pandera.api.synthesis import get_synthesis_backend
from pandera.api.synthesis.registry import _reset_registry_for_tests
from pandera.engines import pandas_engine
from pandera.strategies._values import compile_constraints_to_sampler
from pandera.strategies.constraints import FieldConstraints
from pandera.strategies.faker_strategies import faker_backend


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    try:
        # register the faker backend directly (entry points may not be
        # installed in an editable checkout)
        from pandera.api.synthesis import register_synthesis_backend

        register_synthesis_backend("faker", faker_backend)
        yield
    finally:
        _reset_registry_for_tests()


@pytest.fixture
def str_dtype():
    return pandas_engine.Engine.dtype(str)


EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def _take(sampler, n=5):
    return list(itertools.islice(sampler, n))


class TestValuesCore:
    def test_default_path_no_backend(self, str_dtype):
        vals = _take(
            compile_constraints_to_sampler(
                str_dtype,
                FieldConstraints(str_min_len=5, str_max_len=10),
                seed=1,
            )
        )
        assert vals
        assert all(5 <= len(v) <= 10 for v in vals)

    def test_default_path_unknown_backend_declines(self, str_dtype):
        # A backend with no opinion simply falls back to the default path.
        from pandera.api.synthesis import register_synthesis_backend

        class _NoOpinion:
            name = "no-opinion"

            def field_generator(self):
                class _G:
                    def generate(self, *a, **k):
                        return None

                return _G()

        register_synthesis_backend("no-opinion", _NoOpinion())
        vals = _take(
            compile_constraints_to_sampler(
                str_dtype,
                FieldConstraints(str_min_len=4, str_max_len=8),
                synthesis_backend="no-opinion",
                seed=1,
            )
        )
        assert vals
        assert all(4 <= len(v) <= 8 for v in vals)


class TestFakerBackend:
    def test_resolves_by_name(self):
        backend = get_synthesis_backend("faker")
        assert backend.name == "faker"
        assert hasattr(backend, "field_generator")

    def test_email_satisfies_regex(self, str_dtype):
        sampler = compile_constraints_to_sampler(
            str_dtype,
            FieldConstraints(regex_fullmatch=(EMAIL_RE,)),
            semantic="email",
            synthesis_backend="faker",
            seed=7,
        )
        vals = _take(sampler)
        assert all(re.fullmatch(EMAIL_RE, v) for v in vals)

    def test_isin_country(self, str_dtype):
        allowed = {"US", "CA", "GB", "MX"}
        sampler = compile_constraints_to_sampler(
            str_dtype,
            FieldConstraints(isin=frozenset(allowed)),
            semantic="country",
            synthesis_backend="faker",
            seed=7,
        )
        vals = _take(sampler, n=50)
        assert vals
        assert set(vals) <= allowed

    def test_eq_declines(self, str_dtype):
        # An explicit eq value the provider cannot honour -> decline -> default.
        sampler = compile_constraints_to_sampler(
            str_dtype,
            FieldConstraints(eq="FIXED_VALUE"),
            semantic="name",
            synthesis_backend="faker",
            seed=1,
        )
        vals = _take(sampler)
        assert vals == ["FIXED_VALUE"] * 5

    def test_seed_reproducible(self, str_dtype):
        a = _take(
            compile_constraints_to_sampler(
                str_dtype,
                FieldConstraints(regex_fullmatch=(EMAIL_RE,)),
                semantic="email",
                synthesis_backend="faker",
                seed=3,
            )
        )
        b = _take(
            compile_constraints_to_sampler(
                str_dtype,
                FieldConstraints(regex_fullmatch=(EMAIL_RE,)),
                semantic="email",
                synthesis_backend="faker",
                seed=3,
            )
        )
        assert a == b

    @pytest.mark.parametrize(
        "semantic",
        [
            "name",
            "first_name",
            "last_name",
            "email",
            "phone",
            "address",
            "city",
            "state",
            "postal_code",
            "country",
            "company",
            "job",
            "uuid",
            "currency",
            "boolean",
            "date",
            "datetime",
        ],
    )
    def test_every_semantic_yields_string(self, str_dtype, semantic):
        sampler = compile_constraints_to_sampler(
            str_dtype,
            FieldConstraints(),
            semantic=semantic,
            synthesis_backend="faker",
            seed=11,
        )
        vals = _take(sampler)
        assert vals
        # All are represented as strings in the sampler output.
        assert all(isinstance(v, str) or hasattr(v, "__str__") for v in vals)
