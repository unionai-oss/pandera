"""Pandera type engines."""

import pydantic
from packaging import version


def pydantic_version():
    """Return the pydantic version."""

    return version.parse(pydantic.__version__)


PYDANTIC_V2 = pydantic_version().release >= (2, 0, 0)
