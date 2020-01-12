"""Data validation checks for hypothesis testing."""

from functools import partial
from typing import Callable, Union, Optional, List, Dict

import pandas as pd
from scipy import stats

from . import errors
from .checks import Check, SeriesCheckObj, DataFrameCheckObj


DEFAULT_ALPHA = 0.01


class Hypothesis(Check):
    """Perform a hypothesis test on a Column."""

    RELATIONSHIPS = {
        "greater_than": (lambda stat, pvalue, alpha=DEFAULT_ALPHA:
                         stat > 0 and pvalue / 2 < alpha),
        "less_than": (lambda stat, pvalue, alpha=DEFAULT_ALPHA:
                      stat < 0 and pvalue / 2 < alpha),
        "not_equal": (lambda stat, pvalue, alpha=DEFAULT_ALPHA:
                      pvalue < alpha),
        "equal": (lambda stat, pvalue, alpha=DEFAULT_ALPHA: pvalue >= alpha),
    }

    def __init__(self,
                 test: Callable,
                 samples: Optional[Union[str, List[str]]] = None,
                 groupby: Optional[Union[str, List[str], Callable]] = None,
                 relationship: Union[str, Callable] = "equal",
                 test_kwargs: Dict = None,
                 relationship_kwargs: Dict = None,
                 error: Optional[str] = None):
        """Perform a hypothesis test on a Series or DataFrame.

        Can function on a single column or be grouped by another column.

        :param callable test: A function to check a series schema.
        :param samples: for `Column` or `SeriesSchema` hypotheses, this refers
            to the group keys in the `groupby` column(s) used to group the
            `Series` into a dict of `Series`. The `samples` column(s) are
            passed into the `test` function as positional arguments.

            For `DataFrame`-level hypotheses, `samples` refers to a column or
            multiple columns to pass into the `test` function. The `samples`
            column(s) are passed into the `test`  function as positional
            arguments.
        :param groupby: If a string or list of strings is provided, then these
            columns are used to group the Column Series by `groupby`. If a
            callable is passed, the expected signature is
            DataFrame -> DataFrameGroupby. The function has access to the
            entire dataframe, but the Column.name is selected from this
            DataFrameGroupby object so that a SeriesGroupBy object is passed
            into the `hypothesis_check` function.

            Specifying this argument changes the `fn` signature to:
            dict[str|tuple[str], Series] -> bool|pd.Series[bool]

            Where specific groups can be obtained from the input dict.
        :param relationship: Represents what relationship conditions are
            imposed on the hypothesis test. A function or lambda function can
            be supplied.

            Available built-in relationships are: "greater_than", "less_than",
            "not_equal" or "equal", where "equal" is the null hypothesis.

            If callable, the input function signature should have the signature
            ``(stat: float, pvalue: float, **kwargs)`` where `stat` is the
            hypothesis test statistic, `pvalue` assesses statistical
            significance, and `**kwargs` are other arguments supplied via the
            `**relationship_kwargs` argument.

            Default is "equal" for the null hypothesis.
        :param dict test_kwargs: Key Word arguments to be supplied to the test.
        :param dict relationship_kwargs: Keyword arguments to be supplied to
            the relationship function. e.g. `alpha` could be used to specify a
            threshold in a t-test.
        :param error: error message to show

        :examples:

        Define a two-sample hypothesis test using scipy.

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> from pandera import DataFrameSchema, Column, Hypothesis
        >>> from scipy import stats
        >>>
        >>> schema = DataFrameSchema({
        ...     "height_in_feet": Column(pa.Float, [
        ...         Hypothesis(
        ...             test=stats.ttest_ind,
        ...             samples=["A", "B"],
        ...             groupby="group",
        ...             # assert that the mean height of group "A" is greater
        ...             # than that of group "B"
        ...             relationship=lambda stat, pvalue, alpha=0.1: (
        ...                 stat > 0 and pvalue / 2 < alpha
        ...             ),
        ...             # set alpha criterion to 5%
        ...             relationship_kwargs={"alpha": 0.05}
        ...         )
        ...     ]),
        ...     "group": Column(pa.String),
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

        See :ref:`here<hypothesis>` for more usage details.

        """
        self.test = partial(test, **{} if test_kwargs is None else test_kwargs)
        self.relationship = partial(self._relationships(relationship),
                                    **relationship_kwargs)
        if isinstance(samples, str):
            samples = [samples]
        self.samples = samples
        super(Hypothesis, self).__init__(
            self._hypothesis_check,
            groupby=groupby,
            element_wise=False,
            error=error)

    @property
    def is_one_sample_test(self):
        """Return True if hypothesis is a one-sample test."""
        return len(self.samples) == 1

    def _prepare_series_input(
            self,
            series: pd.Series,
            dataframe_context: pd.DataFrame = None
    ) -> SeriesCheckObj:
        """Prepare Series input for Hypothesis check."""
        self.groups = self.samples
        return super(Hypothesis, self)._prepare_series_input(
            series, dataframe_context)

    def _prepare_dataframe_input(
            self, dataframe: pd.DataFrame) -> DataFrameCheckObj:
        """Prepare input for DataFrameSchema Hypothesis check."""
        if self.groupby is not None:
            raise errors.SchemaDefinitionError(
                "`groupby` cannot be used for DataFrameSchema checks, must "
                "be used in Column checks.")
        if self.is_one_sample_test:
            return dataframe[self.samples[0]]
        check_obj = [(sample, dataframe[sample]) for sample in self.samples]
        return self._format_groupby_input(check_obj, self.samples)

    def _relationships(self, relationship: Union[str, Callable]):
        """Impose a relationship on a supplied Test function.

        :param relationship: represents what relationship conditions are
            imposed on the hypothesis test. A function or lambda function can
            be supplied. If a string is provided, a lambda function will be
            returned from Hypothesis.relationships. Available relationships
            are: "greater_than", "less_than", "not_equal"

        """
        if isinstance(relationship, str):
            if relationship not in self.RELATIONSHIPS:
                raise errors.SchemaError(
                    "The relationship %s isn't a built in method"
                    % relationship)
            relationship = self.RELATIONSHIPS[relationship]
        elif not callable(relationship):
            raise ValueError(
                "expected relationship to be str or callable, found %s" % type(
                    relationship)
            )
        return relationship

    def _hypothesis_check(self, check_obj: Dict[str, pd.Series]):
        """Create a function fn which is checked via the Check parent class.

        :param dict check_obj: a dictionary of pd.Series to be used by
            `_hypothesis_check` and `_vectorized_check`

        """
        if self.is_one_sample_test:
            # one-sample case where no groupby argument supplied, apply to
            # entire column
            return self.relationship(*self.test(check_obj))
        return self.relationship(
            *self.test(*[check_obj.get(s) for s in self.samples]))

    @classmethod
    def two_sample_ttest(
            cls,
            sample1: str,
            sample2: str,
            groupby: Union[str, List[str], callable, None] = None,
            relationship: str = "equal",
            alpha=DEFAULT_ALPHA,
            equal_var=True,
            nan_policy="propagate"):
        """Calculate a t-test for the means of two columns.

        This reuses the scipy.stats.ttest_ind to perfom a two-sided test for
        the null hypothesis that 2 independent samples have identical average
        (expected) values. This test assumes that the populations have
        identical variances by default.

        :param sample1: The first sample group to test. For `Column` and
            `SeriesSchema` hypotheses, refers to the level in the `groupby`
            column. For `DataFrameSchema` hypotheses, refers to column in
            the `DataFrame`.
        :type sample1: str
        :param sample2: The second sample group to test. For `Column` and
            `SeriesSchema` hypotheses, refers to the level in the `groupby`
            column. For `DataFrameSchema` hypotheses, refers to column in
            the `DataFrame`.
        :type sample2: str
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
        :type groupby: str|list[str]|callable|None
        :param relationship: Represents what relationship conditions are
            imposed on the hypothesis test. Available relationships
            are: "greater_than", "less_than", "not_equal", and "equal".
            For example, `group1 greater_than group2` specifies an alternative
            hypothesis that the mean of group1 is greater than group 2 relative
            to a null hypothesis that they are equal.
        :type relationship: str
        :param alpha: (Default value = 0.01) The significance level; the
            probability of rejecting the null hypothesis when it is true. For
            example, a significance level of 0.01 indicates a 1% risk of
            concluding that a difference exists when there is no actual
            difference.
        :type alpha: float
        :param equal_var: (Default value = True) If True (default), perform a
            standard independent 2 sample test that assumes equal population
            variances. If False, perform Welch's t-test, which does not
            assume equal population variance
        :type equal_var: bool
        :param nan_policy: Defines how to handle when input returns nan, one of
            {'propagate', 'raise', 'omit'}, (Default value = 'propagate').
            For more details see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        :type nan_policy: str

        :example:

        The the built-in class method to do a two-sample t-test.

        >>> schema = DataFrameSchema({
        ...     "height_in_feet": Column(
        ...         pa.Float, [
        ...             Hypothesis.two_sample_ttest(
        ...                 sample1="A",
        ...                 sample2="B",
        ...                 groupby="group",
        ...                 relationship="greater_than",
        ...                 alpha=0.05,
        ...                 equal_var=True),
        ...     ]),
        ...     "group": Column(pa.String)
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
        if relationship not in cls.RELATIONSHIPS:
            raise errors.SchemaError(
                "relationship must be one of %s" % set(cls.RELATIONSHIPS))
        return cls(
            test=stats.ttest_ind,
            samples=[sample1, sample2],
            groupby=groupby,
            relationship=relationship,
            test_kwargs={"equal_var": equal_var, "nan_policy": nan_policy},
            relationship_kwargs={"alpha": alpha},
            error="failed two sample ttest between '%s' and '%s'" % (
                sample1, sample2),
        )

    @classmethod
    def one_sample_ttest(
            cls,
            sample: str,
            popmean: float,
            relationship: str,
            alpha: float = DEFAULT_ALPHA):
        """Calculate a t-test for the mean of one column.

        :param sample: The sample group to test. For `Column` and
            `SeriesSchema` hypotheses, refers to the `groupby` level in the
            `Column`. For `DataFrameSchema` hypotheses, refers to column in
            the `DataFrame`.
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

        :example:

        If you want to compare one sample with a pre-defined mean:

        >>> schema = DataFrameSchema({
        ...     "height_in_feet": Column(
        ...         pa.Float, [
        ...             Hypothesis.one_sample_ttest(
        ...                 sample="height_in_feet",
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
        if relationship not in cls.RELATIONSHIPS:
            raise errors.SchemaError(
                "relationship must be one of %s" % set(cls.RELATIONSHIPS))
        return cls(
            test=stats.ttest_1samp,
            samples=sample,
            relationship=relationship,
            test_kwargs={"popmean": popmean},
            relationship_kwargs={"alpha": alpha},
            error="failed one sample ttest for column '%s'" % (sample),
        )
