"""Check backend for pandas."""

from functools import partial
from typing import Dict, List, Optional, Union, cast

import pandas as pd
from multimethod import DispatchError, overload

from pandera.api.base.checks import CheckResult, GroupbyObject
from pandera.api.checks import Check
from pandera.api.pandas.types import (
    is_bool,
    is_field,
    is_table,
    is_table_or_field,
)
from pandera.backends.base import BaseCheckBackend


class PandasCheckBackend(BaseCheckBackend):
    """Check backend ofr pandas."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
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
        raise NotImplementedError

    def aggregate(self, check_obj):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

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
        # NOTE: this behavior should be deprecated such that the user deals with
        # pandas groupby objects instead of dicts.
        if groups is None:
            return {  # type: ignore[return-value]
                (k if isinstance(k, bool) else k[0] if len(k) == 1 else k): v
                for k, v in groupby_obj  # type: ignore[union-attr]
            }
        group_keys = set(
            k[0] if len(k) == 1 else k for k, _ in groupby_obj  # type: ignore[union-attr]
        )
        invalid_groups = [g for g in groups if g not in group_keys]
        if invalid_groups:
            raise KeyError(
                f"groups {invalid_groups} provided in `groups` argument not a "
                f"valid group key. Valid group keys: {group_keys}"
            )
        output = {}
        for group_key, group in groupby_obj:
            if isinstance(group_key, tuple) and len(group_key) == 1:
                group_key = group_key[0]
            if group_key in groups:
                output[group_key] = group

        return output  # type: ignore[return-value]

    @overload
    def preprocess(self, check_obj, key) -> pd.Series:
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Series validation, which has no other context except
        # for the index to groupby on. Right now grouping by the index is not allowed.
        return check_obj

    @overload  # type: ignore [no-redef]
    def preprocess(
        self,
        check_obj: is_field,  # type: ignore [valid-type]
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
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            return check_obj
        return cast(
            Dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj), self.check.groups
            ),
        )

    @overload
    def apply(self, check_obj):
        """Apply the check function to a check object."""
        raise NotImplementedError

    @overload  # type: ignore [no-redef]
    def apply(self, check_obj: dict):
        return self.check_fn(check_obj)

    @overload  # type: ignore [no-redef]
    def apply(self, check_obj: is_field):  # type: ignore [valid-type]
        if self.check.element_wise:
            return check_obj.map(self.check_fn)
        return self.check_fn(check_obj)

    @overload  # type: ignore [no-redef]
    def apply(self, check_obj: is_table):  # type: ignore [valid-type]
        if self.check.element_wise:
            return check_obj.apply(self.check_fn, axis=1)
        return self.check_fn(check_obj)

    @overload
    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        raise TypeError(
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj,
        check_output: is_bool,  # type: ignore [valid-type]
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

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: is_field,  # type: ignore [valid-type]
        check_output: is_field,  # type: ignore [valid-type]
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

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: is_table,  # type: ignore [valid-type]
        check_output: is_field,  # type: ignore [valid-type]
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

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: is_table,  # type: ignore [valid-type]
        check_output: is_table,  # type: ignore [valid-type]
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

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: is_table_or_field,  # type: ignore [valid-type]
        check_output: is_bool,  # type: ignore [valid-type]
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        check_output = bool(check_output)
        return CheckResult(
            check_output,
            check_output,
            check_obj,
            None,
        )

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: dict,
        check_output: is_field,  # type: ignore [valid-type]
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            None,
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
            if exc.__cause__ is not None:
                raise exc.__cause__
            raise exc
        return self.postprocess(check_obj, check_output)
