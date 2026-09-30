"""Pandas backend utilities."""

import re
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


_INLINE_REGEX_FLAGS = (
    (re.IGNORECASE, "i"),
    (re.MULTILINE, "m"),
    (re.DOTALL, "s"),
    (re.VERBOSE, "x"),
)


def regex_inline_flags(pattern: Union[str, re.Pattern]) -> str:
    """Return the flags of a compiled pattern as an inline flag group.

    Backends that take the pattern as a string (e.g. polars) would otherwise
    drop flags such as ``re.IGNORECASE`` set with :func:`re.compile`.
    """
    if not isinstance(pattern, re.Pattern):
        return ""
    flags = "".join(
        letter for flag, letter in _INLINE_REGEX_FLAGS if pattern.flags & flag
    )
    return f"(?{flags})" if flags else ""
