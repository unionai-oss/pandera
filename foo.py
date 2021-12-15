from pandas import DataFrame, Series, Timedelta, Timestamp, concat
from pandas.tseries.offsets import YearEnd
from pandera.typing import Series as PanderaSeries, DataFrame as PanderaDataFrame

def df_in(data_frame: DataFrame) -> None:
    pass
def series_in(series: Series) -> None:
    pass
def pandera_df_in(data_frame: PanderaDataFrame) -> None:
    pass
def pandera_series_in(series: PanderaSeries) -> None:
    pass

df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': list('mnp')})

# YearEnd
Timestamp.now() + YearEnd(1)  # False positive
# String indexes - wrong types

# This is an issue with Slice:
# https://github.com/python/mypy/issues/2410
df.set_index('c').loc['m': 'n']  # False positive


# Timedelta - missing arguments
Timedelta(minutes=2)  # Valid syntax but mypy complains
Timedelta(2, unit='minutes')  # mypy is happy
# How to express this then?
Timedelta(minutes=2, seconds=30)  # Again stubs make mypy complain
Timedelta(2.5, unit='minutes')  # also okay but stubs don't agree
Timedelta(2, unit='minutes') + Timedelta(30, unit='seconds')  # okay, stubs are happy, but ugly and long

# Index is_unique
# reveal_type(df.index.is_unique)
# reveal_type(df['a'].index.is_unique)  # Appears to be a Callable[[], bool] but in fact it is just a bool

# So far we only looked at Pandas problems
# Next, let's look at problems with Pandas+Pandera
#   the function expects a child class, but we pass in an instance of the parent
pandera_series_in(df['a'])  # We should have our Pandera stubs for Pandas and a plugin that knows this kind of stuff
s2 = Series([1, 2, 3])
pandera_series_in(s2)
# we depend on many pandas functions so their output will be defined in terms of pandas classes
#  (disregarding inheritance)

# some pandas functions, like concat, may be typed such that they work with subclasses e.g.
#   concat(dfs: list[TDF]) -> TDF:
# but what about __getattr__ how child classes of DataFrame-s and Series-s vary with each other e.g. how would we type
# the fact that PanderaDataFrame should go with PanderaSeries? overloads, but the stubs can't add overloads for all
# custom child classes

# Similar situation with subclassing
from datetime import datetime
# reveal_type(datetime.min)  # datetime
# Timestamp is a subclass of datatetime
# reveal_type(Timestamp.min)  # datetime

# Interesting issue
# Concat - in some giant environment, i get this behaviour but in a fresh env in which i only installed pandas and
# pandera i don't
concatted = concat([df, df])
reveal_type(concatted)
df_in(concatted)  # False positive
series_in(concatted)  # False negative
