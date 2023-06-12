"""Pandera configuration."""

from enum import Enum
from pydantic import BaseSettings


class ValidationDepth(Enum):
    """Whether to apply checks at schema- or data-level, or both."""

    SCHEMA_ONLY = "SCHEMA_ONLY"
    DATA_ONLY = "DATA_ONLY"
    SCHEMA_AND_DATA = "SCHEMA_AND_DATA"


class PanderaConfig(BaseSettings):
    """Pandera config base class.

    This should pick up environment variables automatically, e.g.:
    export PANDERA_VALIDATION_ENABLED=False
    export PANDERA_VALIDATION_DEPTH=DATA_ONLY
    """

    validation_enabled: bool = True
    validation_depth: ValidationDepth = ValidationDepth.SCHEMA_AND_DATA

    class Config:
        """Pydantic configuration settings."""

        env_prefix = "pandera_"


# this config variable should be accessible globally
CONFIG = PanderaConfig()
