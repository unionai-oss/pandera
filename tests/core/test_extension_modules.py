"""Tests for extension module imports."""

import pytest

from pandera.core.hypotheses import Hypothesis
from pandera.backends.pandas.hypotheses import HAS_SCIPY


def test_hypotheses_module_import() -> None:
    """Test that Hypothesis built-in methods raise import error."""
    if not HAS_SCIPY:
        for fn in [
            lambda: Hypothesis.two_sample_ttest("sample1", "sample2"),  # type: ignore[arg-type]
            lambda: Hypothesis.one_sample_ttest(popmean=10),
        ]:
            with pytest.raises(ImportError):
                fn()
