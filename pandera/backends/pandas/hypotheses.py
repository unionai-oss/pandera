"""Hypothesis backend for pandas."""

from collections.abc import Callable
from functools import partial
from typing import Any, Union, cast

import pandas as pd

from pandera import errors
from pandera.api.hypotheses import Hypothesis
from pandera.api.pandas.types import is_field, is_table
from pandera.backends.pandas.checks import PandasCheckBackend

DEFAULT_ALPHA = 0.01


def greater_than(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for gt hypothesis test."""
    return stat > 0 and pvalue / 2 < alpha


def less_than(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for lt hypothesis test."""
    return stat < 0 and pvalue / 2 < alpha


def not_equal(stat, pvalue, alpha=DEFAULT_ALPHA) -> bool:
    """Evaluate statistic and pvalue for ne hypothesis test."""
    return pvalue < alpha


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

    def preprocess(self, check_obj, key) -> Any:
        if is_table(check_obj) and key is None:
            return self.preprocess_table(check_obj)
        elif is_table(check_obj) and key is not None:
            return self.preprocess_table_with_key(check_obj, key)
        else:
            self.check.groups = self.check.samples  # type: ignore[attr-defined]
            return super().preprocess(check_obj, key)

    def preprocess_table_with_key(
        self,
        check_obj,
        key,
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            return check_obj[key]
        return cast(
            dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj)[key], self.check.groups
            ),
        )

    def preprocess_table(self, check_obj) -> pd.Series:
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
            (sample, check_obj[sample])
            for sample in self.check.samples  # type: ignore[attr-defined]
        ]
        return cast(
            dict[str, pd.DataFrame],
            self._format_groupby_input(check_obj, self.check.groups),
        )
