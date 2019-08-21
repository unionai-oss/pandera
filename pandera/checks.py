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

        return bool_series


class StringMatch(pandera.Check):
    """Check if strings in a pandas.Series match a given regular expression."""
    def __init__(self, regex: str):
        """Create a new StringMatch object based on the given regex.

        :param regex: Regular expression which must be matched
        """
        super().__init__(fn=self.match)
        self.regex = regex

    def match(self, series: pd.Series) -> pd.Series:
        """Check if all strings in the series match the regular expression.

        :returns pd.Series with the comparison result as True or False
        """
        return series.str.match(self.regex)


class StringLength(pandera.Check):
    """Check if the length of strings is within a specified range"""

    def __init__(self, min_len: int = None, max_len: int = None):
        """Create a new StringLength object with a given range

        :param min_len: Minimum length of strings (default: no minimum)
        :param max_len: Maximu length of strings (default: no maximum)
        """
        super().__init__(fn=self.check_string_length)
        self.min_len = min_len
        self.max_len = max_len

    def check_string_length(self, series: pd.Series) -> pd.Series:
        """Check if all strings does have an acceptable length

        :returns pd.Series with the validation result as True or False
        """
        if self.min_len is not None:
            bool_series = series.str.len() >= self.min_len
        else:
            bool_series = pd.Series(data=True, index=series.index)

        if self.max_len is not None:
            return bool_series & (series.str.len() <= self.max_len)
        return bool_series
