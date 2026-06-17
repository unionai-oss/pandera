"""Patch numpy 2 to prevent errors."""

from functools import lru_cache


@lru_cache
def _patch_numpy2():
    """This is a temporary fix for numpy 2.

    pyspark uses np.NaN, which is deprecated in numpy 2.
    """
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
