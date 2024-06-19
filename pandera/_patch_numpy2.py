"""Patch numpy 2 to prevent errors."""


def _patch_numpy2():
    """This is a temporary fix for numpy 2.

    pyspark uses np.NaN, which is deprecated in numpy 2.
    """
    import numpy as np

    np._expired_attrs_2_0.__expired_attributes__.pop("NaN")
    np.NaN = np.nan
