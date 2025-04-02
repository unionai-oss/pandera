# pylint: skip-file
import pandas as pd

pd.Timestamp.now() + pd.tseries.offsets.YearEnd(1)

pd.Timedelta(minutes=2)
pd.Timedelta(2, unit="minutes")

pd.Timedelta(minutes=2, seconds=30)
pd.Timedelta(2.5, unit="minutes")  # mypy error
pd.Timedelta(2, unit="minutes") + pd.Timedelta(30, unit="seconds")
