"""Tests for the semantic data types in :mod:`pandera.dtypes`."""

import pytest

from pandera import dtypes


class TestDecimalScaleGuard:
    """Regression tests for #2496.

    ``Decimal.__init__`` documents that ``scale`` must be between 0 and
    ``precision``, but the guard only checked the upper bound. A negative
    scale slipped through and surfaced later as an unrelated
    ``ValueError: Format specifier missing precision`` from
    ``_scale_to_exp``.
    """

    @pytest.mark.parametrize("scale", [-1, -28])
    def test_negative_scale_raises_documented_error(self, scale):
        with pytest.raises(
            ValueError,
            match=rf"Decimal scale {scale} must be between 0 and 10\.",
        ):
            dtypes.Decimal(precision=10, scale=scale)

    def test_scale_above_precision_raises_documented_error(self):
        with pytest.raises(
            ValueError,
            match=r"Decimal scale 11 must be between 0 and 10\.",
        ):
            dtypes.Decimal(precision=10, scale=11)

    @pytest.mark.parametrize("scale", [0, 5, 10])
    def test_valid_scales_are_accepted(self, scale):
        dtype = dtypes.Decimal(precision=10, scale=scale)
        assert dtype.precision == 10
        assert dtype.scale == scale
