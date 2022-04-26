"""Base functions for Parsing, Validation, and Error Reporting Backends.

This class should implement a common interface of operations needed for
data validation. These operations are exposed as methods that are composed
together to implement the pandera schema specification.
"""

from abc import ABC
from functools import singledispatch
from typing import Optional, Sequence

import pandas as pd

from pandera.checks import Check
from pandera.dtypes import DataType


class BaseSchema(ABC):
    """Abstract base class for a schema backend implementation."""

    def preprocess(
        self,
        check_obj,
        name: str = None,
        inplace: bool = False,
    ):
        pass

    def subsample(
        self,
        check_obj,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        pass

    def validate(
        self,
        check_obj,
        name: Optional[str] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        pass

    def run_check(
        self,
        check_obj,
        schema,
        check,
        check_index: int,
        *args,
    ):
        pass

    def run_checks(
        self,
        check_obj,
        schema,
        checks: Sequence[Check],
        name: Optional[str] = None,
    ):
        pass

    def check_name(self, check_obj, name: Optional[str] = None):
        pass

    def check_nullable(self, check_obj, nullable: bool = False):
        pass

    def check_unique(self, check_obj, unique: bool = False):
        pass

    def check_dtype(self, check_obj, dtype: DataType):
        pass

    def strategy(self):
        pass


class BaseCheck(ABC):
    """Abstract base class for a check backend implementation."""

    def query(self, check_obj):
        """Implements querying behavior to produce subset of check object."""
        pass

    def groupby(self, check_obj):
        """Implements groupby behavior for check object."""
        pass

    def aggregate(self, check_obj):
        """Implements aggregation behavior for check object."""
        pass

    def preprocess(self, check_obj, key):
        """Preprocesses a check object before applying the check function."""
        pass

    def postprocess(self, check_obj):
        """Postprocesses the result of applying the check function."""
        pass

    def __call__(self, check_obj):
        """Apply the check function to a check object."""
        pass

    def statistics(self):
        """Check statistics property."""
        pass

    def strategy(self):
        """Return a data generation strategy."""
        pass
