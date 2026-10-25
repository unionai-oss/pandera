"""Restore numpy 1.x aliases that pyspark < 4 relies on under numpy 2.

pyspark < 4 references aliases that numpy 2 removed (e.g. ``np.NaN``,
``np.float_``). This module re-adds them, but only when pandera is about to
import pyspark *and* the installed pyspark is older than 4.0. It is a no-op
when pyspark isn't installed, when pyspark >= 4 is installed, or on numpy 1.x,
so users who only use the pandas (or any other non-pyspark) API never see a
modified numpy namespace. See https://github.com/unionai-oss/pandera/issues/2134
"""

import re
from functools import lru_cache
from importlib import metadata


def _get_pyspark_version() -> str | None:
    """Return the installed pyspark version string without importing pyspark."""
    try:
        return metadata.version("pyspark")
    except metadata.PackageNotFoundError:
        return None


def _pyspark_needs_numpy2_patch() -> bool:
    """Whether the installed pyspark predates native numpy 2 support (< 4)."""
    pyspark_version = _get_pyspark_version()
    if pyspark_version is None:
        return False
    match = re.match(r"\s*v?(\d+)", pyspark_version)
    if match is None:
        # Unparsable version: don't touch numpy.
        return False
    return int(match.group(1)) < 4


@lru_cache
def _patch_numpy2():
    """Re-add numpy 1.x aliases removed in numpy 2, for pyspark < 4 only.

    Must be called before pandera imports pyspark. Idempotent.
    """
    if not _pyspark_needs_numpy2_patch():
        return

    import numpy as np

    expired_attrs = getattr(np, "_expired_attrs_2_0", None)

    if expired_attrs:
        attrs_replacement = {
            "NaN": np.nan,
            "string_": np.bytes_,
            "float_": np.float64,
            "unicode_": np.str_,
        }
        for attr, replacement in attrs_replacement.items():
            has_attr = expired_attrs.__expired_attributes__.pop(attr, None)
            if has_attr:
                setattr(np, attr, replacement)
