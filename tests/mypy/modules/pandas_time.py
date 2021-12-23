# pylint: skip-file
import pandas as pd

pd.Timestamp.now() + pd.tseries.offsets.YearEnd(
    1
)  # pandas-stubs false positive  # noqa

pd.Timedelta(minutes=2)  # pandas-stubs false positive
pd.Timedelta(2, unit="minutes")  # pandas-stubs false positive

pd.Timedelta(minutes=2, seconds=30)  # pandas-stubs false positive
pd.Timedelta(2.5, unit="minutes")  # pandas-stubs false positive
pd.Timedelta(2, unit="minutes") + pd.Timedelta(30, unit="seconds")
