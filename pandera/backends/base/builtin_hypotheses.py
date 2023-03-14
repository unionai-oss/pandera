# pylint: disable=missing-function-docstring
"""Built-in hypothesis functions base implementation.

This module contains hypothesis function abstract definitions that
correspond to the pandera.api.base.checks.Check methods. These functions do not
actually implement any validation logic and serve as the entrypoint for
dispatching specific implementations based on the data object type, e.g.
`pandas.DataFrame`s.
"""

from typing import Any, Tuple

from pandera.api.hypotheses import Hypothesis


@Hypothesis.register_builtin_check_fn
def two_sample_ttest(
    *samples: Tuple[Any, ...],
    equal_var: bool = True,
    nan_policy: str = "propagate",
):
    raise NotImplementedError


@Hypothesis.register_builtin_check_fn
def one_sample_ttest(
    *samples: Tuple[Any, ...],
    popmean: float,
    nan_policy: str = "propagate",
):
    raise NotImplementedError
