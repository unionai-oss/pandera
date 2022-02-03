# pylint: skip-file
import pandas as pd

pd.Timestamp.now() + pd.tseries.offsets.YearEnd(1)  # false positive

pd.Timedelta(minutes=2)  # false positive
pd.Timedelta(2, unit="minutes")  # false positive

pd.Timedelta(minutes=2, seconds=30)  # false positive
pd.Timedelta(2.5, unit="minutes")  # false positive
pd.Timedelta(2, unit="minutes") + pd.Timedelta(30, unit="seconds")
