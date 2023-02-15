"""Pandas implementation of built-in hypotheses."""

from typing import Tuple

from pandera.backends.pandas.hypotheses import HAS_SCIPY
from pandera.core.extensions import register_hypothesis
from pandera.core.pandas.builtin_checks import PandasData


if HAS_SCIPY:
    from scipy import stats


@register_hypothesis(
    error="failed two sample ttest between '{sample1}' and '{sample2}'",
    # NOTE: an idea here is to use the function signature as a way of
    # inferring the sample name and types, e.g.
    # two_sample_ttest(s1: str, s3: str, *, **kwargs)
    samples_kwtypes={"sample1": str, "sample2": str},
)
def two_sample_ttest(
    *samples: Tuple[PandasData, ...],
    equal_var: bool = True,
    nan_policy: str = "propagate",
) -> Tuple[float, float]:
    """Calculate a t-test for the means of two samples.

    Perform a two-sided test for the null hypothesis that 2 independent
    samples have identical average (expected) values. This test assumes
    that the populations have identical variances by default.

    :param sample1: The first sample group to test. For `Column` and
        `SeriesSchema` hypotheses, refers to the level in the `groupby`
        column. For `DataFrameSchema` hypotheses, refers to column in
        the `DataFrame`.
    :param sample2: The second sample group to test. For `Column` and
        `SeriesSchema` hypotheses, refers to the level in the `groupby`
        column. For `DataFrameSchema` hypotheses, refers to column in
        the `DataFrame`.
    :param groupby: If a string or list of strings is provided, then these
        columns are used to group the Column Series by `groupby`. If a
        callable is passed, the expected signature is
        DataFrame -> DataFrameGroupby. The function has access to the
        entire dataframe, but the Column.name is selected from this
        DataFrameGroupby object so that a SeriesGroupBy object is passed
        into `fn`.

        Specifying this argument changes the `fn` signature to:
        dict[str|tuple[str], Series] -> bool|pd.Series[bool]

        Where specific groups can be obtained from the input dict.
    :param relationship: Represents what relationship conditions are
        imposed on the hypothesis test. Available relationships
        are: "greater_than", "less_than", "not_equal", and "equal".
        For example, `group1 greater_than group2` specifies an alternative
        hypothesis that the mean of group1 is greater than group 2 relative
        to a null hypothesis that they are equal.
    :param alpha: (Default value = 0.01) The significance level; the
        probability of rejecting the null hypothesis when it is true. For
        example, a significance level of 0.01 indicates a 1% risk of
        concluding that a difference exists when there is no actual
        difference.
    :param equal_var: (Default value = True) If True (default), perform a
        standard independent 2 sample test that assumes equal population
        variances. If False, perform Welch's t-test, which does not
        assume equal population variance
    :param nan_policy: Defines how to handle when input returns nan, one of
        {'propagate', 'raise', 'omit'}, (Default value = 'propagate').
        For more details see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

    :example:

    The the built-in class method to do a two-sample t-test.

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> schema = pa.DataFrameSchema({
    ...     "height_in_feet": pa.Column(
    ...         float, [
    ...             pa.Hypothesis.two_sample_ttest(
    ...                 sample1="A",
    ...                 sample2="B",
    ...                 groupby="group",
    ...                 relationship="greater_than",
    ...                 alpha=0.05,
    ...                 equal_var=True),
    ...     ]),
    ...     "group": pa.Column(str)
    ... })
    >>> df = (
    ...     pd.DataFrame({
    ...         "height_in_feet": [8.1, 7, 5.2, 5.1, 4],
    ...         "group": ["A", "A", "B", "B", "B"]
    ...     })
    ... )
    >>> schema.validate(df)[["height_in_feet", "group"]]
        height_in_feet group
    0             8.1     A
    1             7.0     A
    2             5.2     B
    3             5.1     B
    4             4.0     B

    """
    assert (
        len(samples) == 2
    ), "Expected two sample ttest data to contain exactly two samples"
    return stats.ttest_ind(
        samples[0],
        samples[1],
        equal_var=equal_var,
        nan_policy=nan_policy,
    )


@register_hypothesis(
    error="failed one sample ttest for column '{sample}'",
    samples_kwtypes={"sample": str},
)
def one_sample_ttest(
    *samples: Tuple[PandasData, ...],
    popmean: float,
) -> Tuple[float, float]:
    """Calculate a t-test for the mean of one sample.

    :param sample: The sample group to test. For `Column` and
        `SeriesSchema` hypotheses, this refers to the `groupby` level that
        is used to subset the `Column` being checked. For `DataFrameSchema`
        hypotheses, refers to column in the `DataFrame`.
    :param groupby: If a string or list of strings is provided, then these
        columns are used to group the Column Series by `groupby`. If a
        callable is passed, the expected signature is
        DataFrame -> DataFrameGroupby. The function has access to the
        entire dataframe, but the Column.name is selected from this
        DataFrameGroupby object so that a SeriesGroupBy object is passed
        into `fn`.

        Specifying this argument changes the `fn` signature to:
        dict[str|tuple[str], Series] -> bool|pd.Series[bool]

        Where specific groups can be obtained from the input dict.
    :param popmean: population mean to compare `sample` to.
    :param relationship: Represents what relationship conditions are
        imposed on the hypothesis test. Available relationships
        are: "greater_than", "less_than", "not_equal" and "equal". For
        example, `group1 greater_than group2` specifies an alternative
        hypothesis that the mean of group1 is greater than group 2 relative
        to a null hypothesis that they are equal.
    :param alpha: (Default value = 0.01) The significance level; the
        probability of rejecting the null hypothesis when it is true. For
        example, a significance level of 0.01 indicates a 1% risk of
        concluding that a difference exists when there is no actual
        difference.
    :param raise_warning: if True, check raises UserWarning instead of
        SchemaError on validation.

    :example:

    If you want to compare one sample with a pre-defined mean:

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> schema = pa.DataFrameSchema({
    ...     "height_in_feet": pa.Column(
    ...         float, [
    ...             pa.Hypothesis.one_sample_ttest(
    ...                 popmean=5,
    ...                 relationship="greater_than",
    ...                 alpha=0.1),
    ...     ]),
    ... })
    >>> df = (
    ...     pd.DataFrame({
    ...         "height_in_feet": [8.1, 7, 6.5, 6.7, 5.1],
    ...     })
    ... )
    >>> schema.validate(df)
        height_in_feet
    0             8.1
    1             7.0
    2             6.5
    3             6.7
    4             5.1


    """
    assert (
        len(samples) == 1
    ), "Expected one sample ttest data to contain only one sample"
    return stats.ttest_1samp(samples[0], popmean=popmean)
