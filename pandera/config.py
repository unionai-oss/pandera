"""Pandera configuration."""

import os
from enum import Enum

from pydantic import BaseModel


class ValidationDepth(Enum):
    """Whether to apply checks at schema- or data-level, or both."""

    SCHEMA_ONLY = "SCHEMA_ONLY"
    DATA_ONLY = "DATA_ONLY"
    SCHEMA_AND_DATA = "SCHEMA_AND_DATA"


class PanderaConfig(BaseModel):
    """Pandera config base class.

    This should pick up environment variables automatically, e.g.:
    export PANDERA_VALIDATION_ENABLED=False
    export PANDERA_VALIDATION_DEPTH=DATA_ONLY
    """

    validation_enabled: bool = True
    validation_depth: ValidationDepth = ValidationDepth.SCHEMA_AND_DATA


# this config variable should be accessible globally
CONFIG = PanderaConfig(
    validation_enabled=os.environ.get(
        "PANDERA_VALIDATION_ENABLED",
        True,
    ),
    validation_depth=os.environ.get(
        "PANDERA_VALIDATION_DEPTH", ValidationDepth.SCHEMA_AND_DATA
    ),
)
