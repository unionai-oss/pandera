"""Pandera configuration."""

from enum import Enum

from pandera.engines import PYDANTIC_V2

if not PYDANTIC_V2:
    from pydantic import BaseSettings
else:
    try:
        from pydantic_settings import BaseSettings
    except ImportError as exc:
        raise ImportError(
            "If using pydantic v2 you need to install the pydantic_settings "
            "package."
        ) from exc


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
