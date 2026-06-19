"""Pandera configuration."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ValidationDepth(Enum):
    """Whether to apply checks at schema- or data-level, or both."""

    SCHEMA_ONLY = "SCHEMA_ONLY"
    DATA_ONLY = "DATA_ONLY"
    SCHEMA_AND_DATA = "SCHEMA_AND_DATA"


class ValidationScope(Enum):
    """Indicates whether a check/validator operates at a schema of data level."""

    SCHEMA = "schema"
    DATA = "data"


SILENCE_WARNING_PYDANTIC_MODEL = "SILENCE_WARNING_PYDANTIC_MODEL"


@dataclass
class PanderaConfig:
    """Pandera config base class.

    This should pick up environment variables automatically, e.g.:
    export PANDERA_VALIDATION_ENABLED=False
    export PANDERA_VALIDATION_DEPTH=DATA_ONLY
    export PANDERA_CACHE_DATAFRAME=True
    export PANDERA_KEEP_CACHED_DATAFRAME=True
    export PANDERA_USE_NARWHALS_BACKEND=True
    export SILENCE_WARNING_PYDANTIC_MODEL=true

    ``use_narwhals_backend``: when ``True``, Polars, Ibis, and PySpark SQL use
    the Narwhals-powered validation backend. Backends register lazily on first
    schema use; changing this flag via :func:`~pandera.config.set_config`
    re-registers backends that were already registered in the current process.
    """

    validation_enabled: bool = True
    # None is interpreted as "SCHEMA_AND_DATA". None is used as a valid value
    # to support the use case where a pandera validation engine needs to
    # establish default validation depth behavior if the user doesn't explicitly
    # specify the environment variable.
    validation_depth: ValidationDepth | None = None
    cache_dataframe: bool = False
    keep_cached_dataframe: bool = False
    use_narwhals_backend: bool = False
    silenced_warnings: list[str] = field(default_factory=list)

    def is_warning_silenced(self, warning_name: str) -> bool:
        """Check whether a warning is silenced."""
        return warning_name in self.silenced_warnings


_TRUTHY = {"true", "True", "1"}


def _silenced_warnings_from_env() -> list[str]:
    """Collect silenced warnings from environment variables."""
    all_warning_names = [
        SILENCE_WARNING_PYDANTIC_MODEL,
    ]
    return [
        name
        for name in all_warning_names
        if os.environ.get(name, "false") in _TRUTHY
    ]


def _config_from_env_vars():
    validation_enabled = (
        os.environ.get("PANDERA_VALIDATION_ENABLED", "True") in _TRUTHY
    )

    validation_depth = os.environ.get("PANDERA_VALIDATION_DEPTH", None)
    if validation_depth is not None:
        validation_depth = ValidationDepth(validation_depth)

    cache_dataframe = (
        os.environ.get("PANDERA_CACHE_DATAFRAME", "False") in _TRUTHY
    )
    keep_cached_dataframe = (
        os.environ.get("PANDERA_KEEP_CACHED_DATAFRAME", "False") in _TRUTHY
    )
    use_narwhals_backend = (
        os.environ.get("PANDERA_USE_NARWHALS_BACKEND", "False") in _TRUTHY
    )

    return PanderaConfig(
        validation_enabled=validation_enabled,
        validation_depth=validation_depth,
        cache_dataframe=cache_dataframe,
        keep_cached_dataframe=keep_cached_dataframe,
        use_narwhals_backend=use_narwhals_backend,
        silenced_warnings=_silenced_warnings_from_env(),
    )


# this config variable should be accessible globally
CONFIG = _config_from_env_vars()


def _copy_config(config: PanderaConfig) -> PanderaConfig:
    config_copy = copy(config)
    config_copy.silenced_warnings = config.silenced_warnings.copy()
    return config_copy


class _ContextConfig:
    """Context-local config proxy."""

    def __init__(self, config: PanderaConfig) -> None:
        object.__setattr__(
            self,
            "_config",
            ContextVar("pandera_context_config", default=config),
        )

    def get(self) -> PanderaConfig:
        return self._config.get()

    def set(self, config: PanderaConfig):
        return self._config.set(config)

    def reset(self, token) -> None:
        self._config.reset(token)

    def __getattr__(self, name: str):
        return getattr(self.get(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.get(), name, value)


_CONTEXT_CONFIG = _ContextConfig(_copy_config(CONFIG))


def set_config(
    validation_enabled: bool | None = None,
    validation_depth: ValidationDepth | None = None,
    cache_dataframe: bool | None = None,
    keep_cached_dataframe: bool | None = None,
    use_narwhals_backend: bool | None = None,
    silenced_warnings: list[str] | None = None,
) -> None:
    """Set global configuration options.

    Args:
        validation_enabled: Enable or disable validation (default: None)
        validation_depth: Validation depth level (SCHEMA_ONLY, DATA_ONLY, SCHEMA_AND_DATA)
        cache_dataframe: Whether to cache dataframes during validation (default: None)
        keep_cached_dataframe: Whether to keep cached dataframes after validation (default: None)
        use_narwhals_backend: Enable Narwhals-powered backend for compatible backends (default: None)
        silenced_warnings: List of warning names to silence (default: None)

    Note:
        Changing ``use_narwhals_backend`` re-registers Polars, Ibis, and PySpark
        validation backends that were already registered in the current process.
        Backends that have not yet been registered pick up the new value on first
        schema use. See the Narwhals backend documentation for details.
    """
    previous_use_narwhals_backend = CONFIG.use_narwhals_backend

    if validation_enabled is not None:
        CONFIG.validation_enabled = validation_enabled
    if validation_depth is not None:
        CONFIG.validation_depth = validation_depth
    if cache_dataframe is not None:
        CONFIG.cache_dataframe = cache_dataframe
    if keep_cached_dataframe is not None:
        CONFIG.keep_cached_dataframe = keep_cached_dataframe
    if use_narwhals_backend is not None:
        CONFIG.use_narwhals_backend = use_narwhals_backend
    if silenced_warnings is not None:
        CONFIG.silenced_warnings = silenced_warnings

    if (
        use_narwhals_backend is not None
        and use_narwhals_backend != previous_use_narwhals_backend
    ):
        from pandera.backends.narwhals.register import (
            reregister_narwhals_compatible_backends,
        )

        reregister_narwhals_compatible_backends(
            use_narwhals_backend=use_narwhals_backend
        )


@contextmanager
def config_context(
    validation_enabled: bool | None = None,
    validation_depth: ValidationDepth | None = None,
    cache_dataframe: bool | None = None,
    keep_cached_dataframe: bool | None = None,
    use_narwhals_backend: bool | None = None,
    silenced_warnings: list[str] | None = None,
):
    """Temporarily set pandera config options to custom settings."""
    context_config = _copy_config(_CONTEXT_CONFIG.get())

    # Apply new values
    if validation_enabled is not None:
        context_config.validation_enabled = validation_enabled
    if validation_depth is not None:
        context_config.validation_depth = validation_depth
    if cache_dataframe is not None:
        context_config.cache_dataframe = cache_dataframe
    if keep_cached_dataframe is not None:
        context_config.keep_cached_dataframe = keep_cached_dataframe
    if use_narwhals_backend is not None:
        context_config.use_narwhals_backend = use_narwhals_backend
    if silenced_warnings is not None:
        context_config.silenced_warnings = silenced_warnings.copy()

    token = _CONTEXT_CONFIG.set(context_config)
    try:
        yield
    finally:
        _CONTEXT_CONFIG.reset(token)


def reset_config_context(conf: PanderaConfig | None = None):
    """Reset the context configuration to the global configuration."""

    _CONTEXT_CONFIG.set(_copy_config(conf or CONFIG))


def get_config_global() -> PanderaConfig:
    """Get the global configuration."""
    return CONFIG


def get_config_context(
    validation_depth_default: ValidationDepth
    | None = ValidationDepth.SCHEMA_AND_DATA,
) -> PanderaConfig:
    """Gets the configuration context."""
    config = _copy_config(_CONTEXT_CONFIG.get())

    if config.validation_depth is None and validation_depth_default:
        config.validation_depth = validation_depth_default

    return config
