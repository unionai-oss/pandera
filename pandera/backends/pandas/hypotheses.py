"""Hypothesis backend for pandas."""

from functools import partial
from typing import Any, Callable, Dict, Union, cast

import pandas as pd
from multimethod import overload

from pandera import errors
from pandera.api.hypotheses import Hypothesis
from pandera.api.pandas.types import is_field, is_table
from pandera.backends.pandas.checks import PandasCheckBackend

try:
    from scipy import stats  # pylint: disable=unused-import
except ImportError:  # pragma: no cover
    HAS_SCIPY = False
else:
    HAS_SCIPY = True


DEFAULT_ALPHA = 0.01


def greater_than(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for gt hypothesis test."""
    return stat > 0 and pvalue / 2 < alpha


def less_than(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for lt hypothesis test."""
    return stat < 0 and pvalue / 2 < alpha


# pylint: disable=unused-argument
def not_equal(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for ne hypothesis test."""
    return pvalue < alpha


# pylint: disable=unused-argument
def equal(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for eq hypothesis test."""
    return pvalue >= alpha


class PandasHypothesisBackend(PandasCheckBackend):
    """Hypothesis backend implementation for pandas."""

    RELATIONSHIP_FUNCTIONS = {
        "greater_than": greater_than,
        "less_than": less_than,
        "not_equal": not_equal,
        "equal": equal,
    }

    def __init__(self, hypothesis: Hypothesis):
        """Initializes a check backend object."""
        super().__init__(hypothesis)
        assert hypothesis._check_fn is not None, "Check._check_fn must be set."
        self.check = hypothesis

        self.relationship = partial(
            self._relationships(self.check.relationship),
            **(self.check.relationship_kwargs or {}),
        )
        self.check_fn = self._hypothesis_check  # type: ignore [assignment]

    def _relationships(self, relationship: Union[str, Callable]):
        """Impose a relationship on a supplied test function.

        :param relationship: represents what relationship conditions are
            imposed on the hypothesis test. A function or lambda function can
            be supplied. If a string is provided, a lambda function will be
            returned from Hypothesis.relationships. Available relationships
            are: "greater_than", "less_than", "not_equal"

        """
        if isinstance(relationship, str):
            relationship = self.RELATIONSHIP_FUNCTIONS[relationship]
        elif not callable(relationship):
            raise ValueError(
                "expected relationship to be str or callable, found "
                f"{type(relationship)}"
            )
        return relationship

    def _hypothesis_check(self, check_obj):
        """Create a function fn which is checked via the Check parent class.

        :param check_obj: object to validate.
        """
        if is_field(check_obj):
            return self.relationship(
                *self.check._check_fn(check_obj, **self.check._check_kwargs)
            )

        _check_obj = [check_obj.get(s) for s in self.check.samples]
        return self.relationship(
            *self.check._check_fn(*_check_obj, **self.check._check_kwargs)
        )

    @property
    def is_one_sample_test(self):
        """Return True if hypothesis is a one-sample test."""
        return len(self.check.samples) <= 1

    @overload  # type: ignore [no-redef]
    def preprocess(self, check_obj, key) -> Any:
        self.check.groups = self.check.samples
        return super().preprocess(check_obj, key)

    @overload  # type: ignore [no-redef]
    def preprocess(
        self,
        check_obj: is_table,  # type: ignore [valid-type]
        key,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            return check_obj[key]
        return cast(
            Dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj)[key], self.check.groups
            ),
        )

    @overload  # type: ignore [no-redef]
    def preprocess(
        self,
        check_obj: is_table,  # type: ignore [valid-type]
        key: None,
    ) -> pd.Series:
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Series validation, which has no other context except
        # for the index to groupby on. Right now grouping by the index is not allowed.
        if self.check.groupby is not None:
            raise errors.SchemaDefinitionError(
                "`groupby` cannot be used for DataFrameSchema checks, must "
                "be used in Column checks."
            )
        if self.is_one_sample_test:
            return check_obj[self.check.samples[0]]  # type: ignore

        check_obj = [
            (sample, check_obj[sample]) for sample in self.check.samples
        ]
        return cast(
            Dict[str, pd.DataFrame],
            self._format_groupby_input(check_obj, self.check.groups),
        )
