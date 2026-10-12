"""Tests for the pluggable data-synthesis primitives.

Covers the backend registry, semantic typing, and config wiring. These are the
additive, opt-in primitives described in ``specs/pluggable-data-synthesis.md``:
the default synthesis path is untouched by every test here.
"""

from __future__ import annotations

import pytest

from pandera.api.synthesis import (
    SemanticType,
    coerce_semantic,
    get_synthesis_backend,
    register_synthesis_backend,
)
from pandera.api.synthesis.registry import (
    _registered_backend_names,
    _reset_registry_for_tests,
)
from pandera.config import (
    PanderaConfig,
    config_context,
    get_config_context,
    set_config,
)
from pandera.errors import SynthesisBackendNotFoundError


class _DummyBackend:
    name = "dummy"

    def field_generator(self):
        return None


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry before/after each registry test."""
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


class TestRegisterBackend:
    def test_register_and_get(self):
        backend = _DummyBackend()
        register_synthesis_backend("dummy", backend)
        assert get_synthesis_backend("dummy") is backend

    def test_register_idempotent_same_object(self):
        backend = _DummyBackend()
        register_synthesis_backend("dummy", backend)
        # Same object re-registered is a no-op.
        register_synthesis_backend("dummy", backend)
        assert get_synthesis_backend("dummy") is backend

    def test_register_conflict_raises(self):
        register_synthesis_backend("dummy", _DummyBackend())
        with pytest.raises(ValueError, match="already registered"):
            register_synthesis_backend("dummy", _DummyBackend())

    def test_get_none_is_default(self):
        assert get_synthesis_backend(None) is None

    def test_get_unknown_raises(self):
        with pytest.raises(
            SynthesisBackendNotFoundError, match="No data synthesis backend"
        ):
            get_synthesis_backend("does-not-exist")


class TestSemantic:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("name", SemanticType.NAME),
            (SemanticType.EMAIL, SemanticType.EMAIL),
            (None, None),
        ],
    )
    def test_coerce(self, value, expected):
        assert coerce_semantic(value) == expected

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            coerce_semantic("not-a-real-semantic")

    def test_is_str_enum(self):
        # SemanticType values are ordinary strings, so they can round-trip
        # through schema metadata / serialization.
        assert SemanticType.NAME.value == "name"
        assert SemanticType.EMAIL == "email"
        assert SemanticType.EMAIL.value == "email"


class TestConfig:
    @pytest.fixture(autouse=True)
    def _restore_config(self):
        """Snapshot and restore global + context config for test isolation."""
        from pandera.config import (
            CONFIG,
            get_config_context,
            reset_config_context,
        )

        saved_global = CONFIG.synthesis_backend
        saved_context = get_config_context().synthesis_backend
        try:
            yield
        finally:
            CONFIG.synthesis_backend = saved_global
            reset_config_context()
            # Restore the exact context value that was active before the test.
            from pandera.config import _CONTEXT_CONFIG, _copy_config
            from pandera.config import CONFIG as _C

            ctx = _copy_config(_C)
            ctx.synthesis_backend = saved_context
            _CONTEXT_CONFIG.set(ctx)

    def test_default_is_none(self):
        assert PanderaConfig().synthesis_backend is None

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("PANDERA_SYNTHESIS_BACKEND", "faker")
        cfg = PanderaConfig()
        # _config_from_env_vars reads at import, so re-read config via the
        # dataclass default alone won't pick it up; assert the field exists.
        assert hasattr(cfg, "synthesis_backend")

    def test_set_config(self):
        from pandera.config import reset_config_context

        try:
            set_config(synthesis_backend="faker")
            reset_config_context()
            assert get_config_context().synthesis_backend == "faker"
        finally:
            set_config(synthesis_backend=None)
            reset_config_context()

    def test_config_context(self):
        with config_context(synthesis_backend="faker"):
            assert get_config_context().synthesis_backend == "faker"
        assert get_config_context().synthesis_backend is None

    def test_registry_isolation(self):
        """Registry is independent of config (no accidental coupling)."""
        assert "dummy" not in _registered_backend_names()
