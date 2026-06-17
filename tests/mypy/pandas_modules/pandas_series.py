# pylint: skip-file
import pandas as pd


def fn(s: pd.Series[str]) -> bool:
    return True


fn(s=pd.Series([1.0, 1.0, 1.0], dtype=float))  # mypy okay

series = pd.Series([1.0, 1.0, 1.0], dtype=float)
fn(series)  # mypy able to determine `series` type, raises error
