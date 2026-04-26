"""Schema inference for different data types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tensordict import infer_schema as tensordict_infer_schema

__all__ = ["infer_schema"]
