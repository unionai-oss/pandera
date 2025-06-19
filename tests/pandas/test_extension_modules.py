"""Tests for extension module imports."""

import pandas as pd
import pytest

from pandera.api.hypotheses import Hypothesis


try:
    from scipy import stats  # pylint: disable=unused-import
except ImportError:  # pragma: no cover
    SCIPY_INSTALLED = False
else:
    SCIPY_INSTALLED = True


def test_hypotheses_module_import() -> None:
    """Test that Hypothesis built-in methods raise import error."""
    data = pd.Series([1, 2, 3])
    if not SCIPY_INSTALLED:
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
