"""Check backend for pyspark."""

from functools import partial
from typing import Dict, List, Optional

from multimethod import DispatchError, overload
from pyspark.sql import DataFrame

from pandera.api.base.checks import CheckResult, GroupbyObject
from pandera.api.checks import Check
from pandera.api.pyspark.types import (
    PysparkDataframeColumnObject,
    is_bool,
    is_table,
)
from pandera.backends.base import BaseCheckBackend


class PySparkCheckBackend(BaseCheckBackend):
    """Check backend for PySpark."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: DataFrame):  # pragma: no cover
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
    ) -> Dict[str, DataFrame]:  # pragma: no cover
        raise NotImplementedError

    @overload  # type: ignore [no-redef]
    def preprocess(
        self,
        check_obj: DataFrame,
        key: str,  # type: ignore [valid-type]
    ) -> DataFrame:
        return check_obj

    # Workaround for multimethod not supporting Optional arguments
    # such as `key: Optional[str]` (fails in multimethod)
    # https://github.com/coady/multimethod/issues/90
    # FIXME when the multimethod supports Optional args # pylint: disable=fixme
    @overload  # type: ignore [no-redef]
    def preprocess(
        self,
        check_obj: DataFrame,  # type: ignore [valid-type]
    ) -> DataFrame:
        return check_obj

    @overload
    def apply(self, check_obj):
        """Apply the check function to a check object."""
        raise NotImplementedError

    @overload  # type: ignore [no-redef]
    def apply(self, check_obj: DataFrame):
        return self.check_fn(check_obj)  # pragma: no cover

    @overload  # type: ignore [no-redef]
    def apply(self, check_obj: is_table):  # type: ignore [valid-type]
        return self.check_fn(check_obj)  # pragma: no cover

    @overload  # type: ignore [no-redef]
    def apply(self, check_obj: DataFrame, column_name: str, kwargs: dict):  # type: ignore [valid-type]
        # kwargs['column_name'] = column_name
        # return self.check._check_fn(check_obj, *list(kwargs.values()))
        check_obj_and_col_name = PysparkDataframeColumnObject(
            check_obj, column_name
        )
        return self.check._check_fn(check_obj_and_col_name, **kwargs)

    @overload
    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        raise TypeError(  # pragma: no cover
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

    def __call__(
        self,
        check_obj: DataFrame,
        key: Optional[str] = None,
    ) -> CheckResult:
        if key is None:
            # pylint:disable=no-value-for-parameter
            check_obj = self.preprocess(check_obj)
        else:
            check_obj = self.preprocess(check_obj, key)

        try:
            if key is None:
                check_output = self.apply(check_obj)
            else:
                check_output = (
                    self.apply(  # pylint:disable=too-many-function-args
                        check_obj, key, self.check._check_kwargs
                    )
                )

        except DispatchError as exc:  # pragma: no cover
            if exc.__cause__ is not None:
                raise exc.__cause__
            raise exc
        except TypeError as err:
            raise err
        return self.postprocess(check_obj, check_output)
