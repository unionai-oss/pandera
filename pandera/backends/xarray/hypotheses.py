"""Hypothesis backend for xarray."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Union

from pandera import errors
from pandera.api.hypotheses import Hypothesis
from pandera.backends.xarray.checks import XarrayCheckBackend

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


class XarrayHypothesisBackend(XarrayCheckBackend):
    """Hypothesis checks on xarray containers."""

    RELATIONSHIP_FUNCTIONS = {
        "greater_than": greater_than,
        "less_than": less_than,
        "not_equal": not_equal,
        "equal": equal,
    }

    def __init__(self, hypothesis: Hypothesis):
        super().__init__(hypothesis)
        self.check = hypothesis
        self.relationship = partial(
            self._relationships(self.check.relationship),
            **(self.check.relationship_kwargs or {}),
        )
        self.check_fn = self._hypothesis_check  # type: ignore[assignment]

    def _relationships(self, relationship: Union[str, Callable]):
        if isinstance(relationship, str):
            relationship = self.RELATIONSHIP_FUNCTIONS[relationship]
        elif not callable(relationship):
            raise ValueError(
                "expected relationship to be str or callable, found "
                f"{type(relationship)}"
            )
        return relationship

    def _hypothesis_check(self, check_obj: Any):
        import xarray as xr

        if isinstance(check_obj, xr.DataArray):
            return self.relationship(
                *self.check._check_fn(
                    check_obj,
                    **getattr(self.check, "_check_kwargs", {}),
                )
            )
        raise errors.SchemaDefinitionError(
            "multi-sample hypothesis tests on xarray.Dataset are not "
            "implemented."
        )
