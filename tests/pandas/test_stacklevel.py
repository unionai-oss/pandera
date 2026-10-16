"""Regression tests for warning attribution via ``stacklevel``.

https://github.com/unionai-oss/pandera/issues/1896
"""

import inspect
import warnings
from pathlib import Path

import pandas as pd

import pandera.pandas as pa


def test_regex_dtypes_warning_points_at_caller():
    """Warnings raised by a pandera API call must be attributed to the code
    that called pandera, not to pandera's internal ``warnings.warn`` line."""
    schema = pa.DataFrameSchema({"col*": pa.Column(str, regex=True)})
    df = pd.DataFrame({"col": ["a"], "col2": ["b"]})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        expected_lineno = inspect.currentframe().f_lineno + 1
        schema.dtypes  # user call site; must be the reported warning origin
        assert not df.empty  # keep ``df`` used after the attributed line
    regex_warnings = [w for w in record if "regex" in str(w.message).lower()]
    assert regex_warnings, f"expected a regex warning, got {record!r}"
    warning = regex_warnings[0]
    assert Path(warning.filename).resolve() == Path(__file__).resolve(), (
        f"warning attributed to {warning.filename}:{warning.lineno}, "
        f"expected {Path(__file__).resolve()}:{expected_lineno}"
    )
    assert warning.lineno == expected_lineno
