"""Pandas backend utilities."""

from typing import Union

from pandera.dtypes import UniqueSettings


def convert_uniquesettings(unique: UniqueSettings) -> Union[bool, str]:
    """
    Converts UniqueSettings object to string that can be passed onto pandas .duplicated() call
    """
    # Default `keep` argument for pandas .duplicated() function
    keep_argument: Union[bool, str]
    if unique == "exclude_first":
        keep_argument = "first"
    elif unique == "exclude_last":
        keep_argument = "last"
    elif unique == "all":
        keep_argument = False
    else:
        raise ValueError(
            str(unique) + " is not a recognized report_duplicates value"
        )
    return keep_argument
