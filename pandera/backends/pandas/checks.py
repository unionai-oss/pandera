"""Check backend for pandas."""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from multimethod import multimethod, DispatchError

from pandera.backends.base import BaseCheckBackend
from pandera.core.base.checks import CheckResult
from pandera.core.checks import Check

GroupbyObject = Union[
    pd.core.groupby.SeriesGroupBy, pd.core.groupby.DataFrameGroupBy
]


class PandasCheckBackend(BaseCheckBackend):
    def __init__(self, check: Check):
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: Union[pd.Series, pd.DataFrame]):
        """Implements groupby behavior for check object."""
        assert self.check.groupby is not None, "Check.groupby must be set."
        if isinstance(self.check.groupby, (str, list)):
            return check_obj.groupby(self.check.groupby)
        return self.check.groupby(check_obj)

    def query(self, check_obj):
        """Implements querying behavior to produce subset of check object."""
        # TODO
        ...

    def aggregate(self, check_obj):
        """Implements aggregation behavior for check object."""
        # TODO
        ...

    @staticmethod
    def _format_groupby_input(
        groupby_obj: GroupbyObject,
        groups: Optional[List[str]],
    ) -> Union[Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
        """Format groupby object into dict of groups to Series or DataFrame.

        :param groupby_obj: a pandas groupby object.
        :param groups: only include these groups in the output.
        :returns: dictionary mapping group names to Series or DataFrame.
        """
        # TODO: this behavior should be deprecated such that the user deals with pandas
        # groupby objects instead of dicts.
        if groups is None:
            return dict(list(groupby_obj))
        group_keys = set(group_key for group_key, _ in groupby_obj)
        invalid_groups = [g for g in groups if g not in group_keys]
        if invalid_groups:
            raise KeyError(
                f"groups {invalid_groups} provided in `groups` argument not a valid group "
                f"key. Valid group keys: {group_keys}"
            )
        return {
            group_key: group
            for group_key, group in groupby_obj
            if group_key in groups
        }

    @multimethod
    def preprocess(self, check_obj, key) -> pd.Series:
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Series validation, which has no other context except
        # for the index to groupby on. Right now grouping by the index is not allowed.
        return check_obj

    @preprocess.register
    def _(
        self,
        check_obj: pd.Series,
        key,
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        if self.check.groupby is None:
            return check_obj
        return cast(
            Dict[str, pd.Series],
            self._format_groupby_input(
                self.groupby(check_obj), self.check.groups
            ),
        )

    @preprocess.register
    def _(
        self,
        check_obj: pd.DataFrame,
        key: Union[str, tuple],
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            return check_obj[key]
        return cast(
            Dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj)[key], self.check.groups
            ),
        )

    @preprocess.register
    def _(
        self,
        check_obj: pd.DataFrame,
        key: None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            return check_obj
        return cast(
            Dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj), self.check.groups
            ),
        )

    @multimethod
    def apply(self, check_obj):
        """Apply the check function to a check object."""
        raise NotImplementedError

    @apply.register
    def _(self, check_obj: dict):
        return self.check_fn(check_obj)

    @apply.register
    def _(self, check_obj: pd.Series):
        if self.check.element_wise:
            return check_obj.map(self.check_fn)
        return self.check_fn(check_obj)

    @apply.register
    def _(self, check_obj: pd.DataFrame):
        if self.check.element_wise:
            return check_obj.apply(self.check_fn, axis=1)
        return self.check_fn(check_obj)

    @multimethod
    def postprocess(
        self,
        check_obj: Any,
        check_output: Union[bool, np.bool_],
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output=check_output,
            check_passed=check_output,
            checked_object=check_obj,
            failure_cases=None,
        )

    def _get_series_failure_cases(
        self, check_obj, check_output: pd.Series
    ) -> Optional[pd.Series]:
        if not check_obj.index.equals(check_output.index):
            return None

        failure_cases = check_obj[~check_output]
        if not failure_cases.empty and self.check.n_failure_cases is not None:
            # NOTE: this is a hack to support pyspark.pandas and modin, since you
            # can't use groupby on a dataframe with another dataframe
            if type(failure_cases).__module__.startswith(
                ("pyspark.pandas", "modin.pandas")
            ):
                failure_cases = (
                    failure_cases.rename("failure_cases")
                    .to_frame()
                    .assign(check_output=check_output)
                    .groupby("check_output")
                    .head(self.check.n_failure_cases)["failure_cases"]
                )
            else:
                failure_cases = failure_cases.groupby(check_output).head(
                    self.check.n_failure_cases
                )
        return failure_cases

    @postprocess.register
    def _(
        self,
        check_obj: pd.Series,
        check_output: pd.Series,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        if check_obj.index.equals(check_output.index) and self.check.ignore_na:
            check_output = check_output | check_obj.isna()
        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            self._get_series_failure_cases(check_obj, check_output),
        )

    @postprocess.register
    def _(
        self,
        check_obj: pd.DataFrame,
        check_output: pd.Series,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        if check_obj.index.equals(check_output.index) and self.check.ignore_na:
            check_output = check_output | check_obj.isna().all(axis="columns")
        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            self._get_series_failure_cases(check_obj, check_output),
        )

    @postprocess.register
    def _(
        self,
        check_obj: pd.DataFrame,
        check_output: pd.DataFrame,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        assert check_obj.shape == check_output.shape
        check_obj = check_obj.unstack()
        check_output = check_output.unstack()
        if check_obj.index.equals(check_output.index) and self.check.ignore_na:
            check_output = check_output | check_obj.isna()
        failure_cases = (
            check_obj[~check_output]  # type: ignore  [call-overload]
            .rename("failure_case")
            .rename_axis(["column", "index"])
            .reset_index()
        )
        if not failure_cases.empty and self.check.n_failure_cases is not None:
            failure_cases = failure_cases.drop_duplicates().head(
                self.check.n_failure_cases
            )
        return CheckResult(
            check_output,
            check_output.all(axis=None),  # type: ignore
            check_obj,
            failure_cases,
        )

    @postprocess.register
    def _(
        self,
        check_obj: Union[pd.Series, pd.DataFrame],
        check_output: Union[bool, np.bool_],
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        check_output = bool(check_output)
        return CheckResult(
            check_output,
            check_output,
            check_obj,
            None,
        )

    @postprocess.register
    def _(self, check_obj: dict, check_output: pd.Series):
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            None,
        )

    @postprocess.register
    def _(self, check_obj: Any, check_output: Any):
        """Postprocesses the result of applying the check function."""
        raise TypeError(
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    def __call__(
        self,
        check_obj: Union[pd.Series, pd.DataFrame],
        key: Optional[str] = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        try:
            check_output = self.apply(check_obj)
        except DispatchError as exc:
            raise exc.__cause__
        return self.postprocess(check_obj, check_output)
