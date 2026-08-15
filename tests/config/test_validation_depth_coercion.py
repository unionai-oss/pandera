"""
Regression test for validation_depth string coercion (closes #2431).

Before the fix, config_context(validation_depth="SCHEMA_ONLY") stored
the raw string instead of coercing to ValidationDepth enum. Since
ValidationDepth is not a str-Enum, all @validate_scope comparisons
(config.validation_depth == ValidationDepth.X) were silently False,
disabling all depth gating.
"""

import pytest

from pandera.config import (
    CONFIG,
    ValidationDepth,
    config_context,
    get_config_context,
    set_config,
)


class TestValidationDepthCoercion:
    """String validation_depth must be coerced to ValidationDepth enum."""

    def test_config_context_coerces_string(self):
        """config_context with string value must produce enum."""
        with config_context(validation_depth="SCHEMA_ONLY"):
            config = get_config_context()
            assert config.validation_depth == ValidationDepth.SCHEMA_ONLY
            assert isinstance(config.validation_depth, ValidationDepth)

    def test_config_context_accepts_enum(self):
        """config_context with enum value must work as before."""
        with config_context(validation_depth=ValidationDepth.SCHEMA_ONLY):
            config = get_config_context()
            assert config.validation_depth == ValidationDepth.SCHEMA_ONLY

    def test_set_config_coerces_string(self):
        """set_config with string value coerces to enum."""
        original = CONFIG.validation_depth
        try:
            set_config(validation_depth="DATA_ONLY")
            assert CONFIG.validation_depth == ValidationDepth.DATA_ONLY
            assert isinstance(CONFIG.validation_depth, ValidationDepth)
        finally:
            CONFIG.validation_depth = original

    def test_invalid_string_raises(self):
        """Invalid string must raise ValueError."""
        with pytest.raises(ValueError):
            with config_context(validation_depth="INVALID_DEPTH"):
                pass
