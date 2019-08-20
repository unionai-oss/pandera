"""Built-in standard checks for pandera"""

import pandera
import pandas as pd


class ValueRange(pandera.Check):
    """Check whether values are within a certain range."""
    def __init__(self, min_value=None, max_value=None):
        """Create a new ValueRange check object.

        :param min_value: Allowed minimum value. Should be a type comparable to
            the type of the pandas series to be validated (e.g. a numerical type
            for float or int and a datetime for datetime) .
        :param max_value: Allowed maximum value. Should be a type comparable to
            the type of the pandas series to be validated (e.g. a numerical type
            for float or int and a datetime for datetime).
        """
        super().__init__(fn=self.check)
        self.min_value = min_value
        self.max_value = max_value

    def check(self, series: pd.Series) -> pd.Series:
        """Compare the values of the series to the predefined limits.

        :returns pd.Series with the comparison result as True or False
        """
        if self.min_value is not None:
            bool_series = series >= self.min_value
        else:
            bool_series = pd.Series(data=True, index=series.index)

        if self.max_value is not None:
            return bool_series & (series <= self.max_value)
        else:
            return bool_series
