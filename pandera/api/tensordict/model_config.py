"""Class-based TensorDict model API configuration."""

from typing import Any

from pandera.api.dataframe.model_config import BaseConfig as _BaseConfig


class BaseConfig(_BaseConfig):
    """Define TensorDictSchema-wide options.

    *new in 0.19.0*
    """

    batch_size: tuple[int, ...] | None = None
