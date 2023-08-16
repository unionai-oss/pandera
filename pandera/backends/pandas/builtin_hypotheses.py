# pylint: disable=missing-function-docstring
"""Pandas implementation of built-in hypotheses."""

from typing import Tuple

from pandera.api.extensions import register_builtin_hypothesis
from pandera.backends.pandas.builtin_checks import PandasData


@register_builtin_hypothesis(
    error="failed two sample ttest between '{sample1}' and '{sample2}'",
    samples_kwtypes={"sample1": str, "sample2": str},
)
def two_sample_ttest(
    *samples: Tuple[PandasData, ...],
    equal_var: bool = True,
    nan_policy: str = "propagate",
) -> Tuple[float, float]:
    from scipy import stats  # pylint: disable=import-outside-toplevel

    assert (
        len(samples) == 2
    ), "Expected two sample ttest data to contain exactly two samples"
    return stats.ttest_ind(
        samples[0],
        samples[1],
        equal_var=equal_var,
        nan_policy=nan_policy,
    )


@register_builtin_hypothesis(
    error="failed one sample ttest for column '{sample}'",
    samples_kwtypes={"sample": str},
)
def one_sample_ttest(
    *samples: Tuple[PandasData, ...],
    popmean: float,
    nan_policy: str = "propagate",
) -> Tuple[float, float]:
    from scipy import stats  # pylint: disable=import-outside-toplevel

    assert (
        len(samples) == 1
    ), "Expected one sample ttest data to contain only one sample"
    return stats.ttest_1samp(
        samples[0], popmean=popmean, nan_policy=nan_policy
    )
