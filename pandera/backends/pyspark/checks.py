"""Check backend for pyspark."""

from functools import partial
from typing import Dict, List, Optional, Union

from pandera.api.base.checks import CheckResult, GroupbyObject
from pandera.api.checks import Check
from pandera.api.pyspark.types import (
    PysparkDataframeColumnObject,
    is_bool,
    is_table,
)
from pandera.backends.base import BaseCheckBackend
from pandera.api.pyspark.types import DataFrameTypes


class PySparkCheckBackend(BaseCheckBackend):
    """Check backend for PySpark."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: DataFrameTypes):  # pragma: no cover
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
    ) -> Dict[str, DataFrameTypes]:  # pragma: no cover
        raise NotImplementedError

    def preprocess(
        self,
        check_obj: DataFrameTypes,
        key: str,  # type: ignore [valid-type]
    ) -> DataFrameTypes:
        return check_obj

    def apply(
        self,
        check_obj: Union[DataFrameTypes, is_table],
        column_name: str = None,
        kwargs: dict = None,
    ):
        if column_name and kwargs:
            check_obj_and_col_name = PysparkDataframeColumnObject(
                check_obj, column_name
            )
            return self.check._check_fn(check_obj_and_col_name, **kwargs)

        else:
            return self.check_fn(check_obj)  # pragma: no cover

    def postprocess(
        self,
        check_obj: DataFrameTypes,
        check_output: is_bool,  # type: ignore [valid-type]
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output=check_output,
            check_passed=check_output,
            checked_object=check_obj,
            failure_cases=None,
        )

    def __call__(
        self,
        check_obj: DataFrameTypes,
        key: Optional[str] = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)

        check_output = self.apply(  # pylint:disable=too-many-function-args
            check_obj, key, self.check._check_kwargs
        )

        return self.postprocess(check_obj, check_output)
