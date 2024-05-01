"""Pandera configuration."""

import os
from contextlib import contextmanager
from copy import deepcopy
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
    # None is interpreted as "SCHEMA_AND_DATA". None is used as a valid value
    # to support the use case where a pandera validation engine needs to
    # establish default validation depth behavior if the user doesn't explicitly
    # specify the environment variable.
    validation_depth: Optional[ValidationDepth] = None
    cache_dataframe: bool = False
    keep_cached_dataframe: bool = False


# this config variable should be accessible globally
CONFIG = PanderaConfig(
    validation_enabled=os.environ.get(
        "PANDERA_VALIDATION_ENABLED",
        True,
    ),
    validation_depth=os.environ.get(
        "PANDERA_VALIDATION_DEPTH",
        None,
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

_CONTEXT_CONFIG = deepcopy(CONFIG)


@contextmanager
def config_context(
    validation_enabled: Optional[bool] = None,
    validation_depth: Optional[ValidationDepth] = None,
    cache_dataframe: Optional[bool] = None,
    keep_cached_dataframe: Optional[bool] = None,
):
    """Temporarily set pandera config options to custom settings."""
    _outer_config_ctx = get_config_context(validation_depth_default=None)

    try:
        if validation_enabled is not None:
            _CONTEXT_CONFIG.validation_enabled = validation_enabled
        if validation_depth is not None:
            _CONTEXT_CONFIG.validation_depth = validation_depth
        if cache_dataframe is not None:
            _CONTEXT_CONFIG.cache_dataframe = cache_dataframe
        if keep_cached_dataframe is not None:
            _CONTEXT_CONFIG.keep_cached_dataframe = keep_cached_dataframe

        yield
    finally:
        reset_config_context(_outer_config_ctx)


def reset_config_context(conf: Optional[PanderaConfig] = None):
    """Reset the context configuration to the global configuration."""
    # pylint: disable=global-statement
    global _CONTEXT_CONFIG
    _CONTEXT_CONFIG = deepcopy(conf or CONFIG)


def get_config_global() -> PanderaConfig:
    """Get the global configuration."""
    return CONFIG


def get_config_context(
    validation_depth_default: Optional[
        ValidationDepth
    ] = ValidationDepth.SCHEMA_AND_DATA,
) -> PanderaConfig:
    """Gets the configuration context."""
    config = deepcopy(_CONTEXT_CONFIG)

    if config.validation_depth is None and validation_depth_default:
        config.validation_depth = validation_depth_default

    return config
