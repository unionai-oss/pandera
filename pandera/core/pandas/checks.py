import inspect
import operator
import re
from collections import ChainMap, namedtuple
from functools import partial, wraps
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

import pandas as pd

import pandera.errors as errors
import pandera.strategies as st
from pandera.core import base
from pandera.core.pandas.types import is_field, is_table, is_index, is_multiindex


CheckResult = namedtuple(
    "CheckResult",
    ["check_output", "check_passed", "checked_object", "failure_cases"],
)


GroupbyObject = Union[
    pd.core.groupby.SeriesGroupBy, pd.core.groupby.DataFrameGroupBy
]


# TODO: consider moving checks out of the `core.pandas` module so that it has a different
# API altogether. Basically the API would have 4 layers of interaction:
#
# 1. simply using the Check.built_in_method or Check(lambda ...) API
# 2. registering custom checks via the extensions API.
# 3. extending built-in checks to support other types (e.g. non-pandas data containers).
# 4. adding additional backend operations to the Check API to support other types.
#    - querying a subset of the check object
#    - groupby
#    - aggregate
#    - preprocess
#    - postprocess
#    - __call__
#    - strategy

# TODO: This module should implement the pandas-specific check specification


class BaseCheck(base.BaseCheck):
    ...


class Check(BaseCheck):
    ...


class Hypothesis(BaseCheck):
    ...
