"""Pandera configuration."""


import os
from contextlib import contextmanager
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ValidationDepth(Enum):
    """Whether to apply checks at schema- or data-level, or both."""

    SCHEMA_ONLY = "SCHEMA_ONLY"
    DATA_ONLY = "DATA_ONLY"
    SCHEMA_AND_DATA = "SCHEMA_AND_DATA"


class ValidationScope(Enum):
    """Indicates whether a check/validator operates at a schema of data level."""

    SCHEMA = "schema"
    DATA = "data"


class PanderaConfig(BaseModel):
    """Pandera config base class.

    This should pick up environment variables automatically, e.g.:
    export PANDERA_VALIDATION_ENABLED=False
    export PANDERA_VALIDATION_DEPTH=DATA_ONLY
    export PANDERA_CACHE_DATAFRAME=True
    export PANDERA_KEEP_CACHED_DATAFRAME=True
    """

    validation_enabled: bool = True
    validation_depth: ValidationDepth = ValidationDepth.SCHEMA_AND_DATA
    cache_dataframe: bool = False
    keep_cached_dataframe: bool = False


# this config variable should be accessible globally
CONFIG = PanderaConfig(
    validation_enabled=os.environ.get(
        "PANDERA_VALIDATION_ENABLED",
        True,
    ),
    validation_depth=os.environ.get(
        "PANDERA_VALIDATION_DEPTH", ValidationDepth.SCHEMA_AND_DATA
    ),
    cache_dataframe=os.environ.get(
        "PANDERA_CACHE_DATAFRAME",
        False,
    ),
    keep_cached_dataframe=os.environ.get(
        "PANDERA_KEEP_CACHED_DATAFRAME",
        False,
    ),
)


@contextmanager
def config_context(
    validation_enabled: Optional[bool] = None,
    validation_depth: Optional[ValidationDepth] = None,
    cache_dataframe: Optional[bool] = None,
    keep_cached_dataframe: Optional[bool] = None,
):
    """Temporarily set pandera config options to custom settings."""
    global CONFIG

    original_config = CONFIG.model_copy()

    # if validation_enabled is not None:
    #     CONFIG.validation_enabled = validation_enabled
    # if validation_depth is not None:
    #     CONFIG.validation_depth = validation_depth
    # if cache_dataframe is not None:
    #     CONFIG.cache_dataframe = cache_dataframe
    # if keep_cached_dataframe is not None:
    #     CONFIG.keep_cached_dataframe = keep_cached_dataframe

    yield

    CONFIG = original_config
