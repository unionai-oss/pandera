"""Pandera type engines."""

from pandera.engines.utils import pydantic_version


PYDANTIC_V2 = pydantic_version().release >= (2, 0, 0)
