"""Tests for extension module imports."""

import pytest

import pandas as pd

from pandera.api.hypotheses import Hypothesis
from pandera.backends.pandas.hypotheses import HAS_SCIPY


def test_hypotheses_module_import() -> None:
    """Test that Hypothesis built-in methods raise import error."""
    data = pd.Series([1, 2, 3])
    if not HAS_SCIPY:
        for fn, check_args in [
            (
                lambda: Hypothesis.two_sample_ttest("sample1", "sample2"),
                pd.DataFrame({"sample1": data, "sample2": data}),
            ),
            (lambda: Hypothesis.one_sample_ttest(popmean=10), data),
        ]:
            with pytest.raises(ImportError):
                check = fn()
                check(check_args)
