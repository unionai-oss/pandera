"""Data validation checks for hypothesis testing."""

from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional, Union

from pandera import errors
from pandera.core.checks import Check
from pandera.strategies import SearchStrategy


class Hypothesis(Check):
    """Special type of :class:`Check` that defines hypothesis tests on data."""

    RELATIONSHIPS = {"greater_than", "less_than", "not_equal", "equal"}

    # pylint: disable=too-many-locals
    def __init__(
        self,
        test: Callable,
        samples: Optional[Union[str, List[str]]] = None,
        groupby: Optional[Union[str, List[str], Callable]] = None,
        relationship: Union[str, Callable] = "equal",
        alpha: Optional[float] = None,
        test_kwargs: Dict = None,
        relationship_kwargs: Dict = None,
        name: Optional[str] = None,
        error: Optional[str] = None,
        raise_warning: bool = False,
        n_failure_cases: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        statistics: Dict[str, Any] = None,
        strategy: Optional[SearchStrategy] = None,
        **check_kwargs,
    ) -> None:
        """Perform a hypothesis test on a Series or DataFrame.

        :param test: The hypothesis test function. It should take one or more
            arrays as positional arguments and return a test statistic and a
            p-value. The arrays passed into the test function are determined
            by the ``samples`` argument.
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
        :param alpha: significance level, if applicable to the hypothesis check.
        :param dict test_kwargs: Keyword arguments to be supplied to the test.
        :param dict relationship_kwargs: Keyword arguments to be supplied to
            the relationship function. e.g. `alpha` could be used to specify a
            threshold in a t-test.
        :param name: optional name of hypothesis test
        :param error: error message to show
        :param raise_warning: if True, raise a UserWarning and do not throw
            exception instead of raising a SchemaError for a specific check.
            This option should be used carefully in cases where a failing
            check is informational and shouldn't stop execution of the program.
        :param n_failure_cases: report the first n unique failure cases. If
            None, report all failure cases.
        :param title: A human-readable label for the check.
        :param description: An arbitrary textual description of the check.
        :param statistics: kwargs to pass into the check function. These values
            are serialized and represent the constraints of the checks.
        :param strategy: A hypothesis strategy, used for implementing data
            synthesis strategies for this check.
        :param check_kwargs: key-word arguments to pass into ``check_fn``

        :examples:

        Define a two-sample hypothesis test using scipy.

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> from scipy import stats
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "height_in_feet": pa.Column(float, [
        ...         pa.Hypothesis(
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
        ...     "group": pa.Column(str),
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
        if (
            not callable(relationship)
            and relationship not in self.RELATIONSHIPS
        ):
            raise errors.SchemaInitError(
                f"The relationship {relationship} isn't a built in method"
            )

        self.test = partial(test, **{} if test_kwargs is None else test_kwargs)
        update_wrapper(self.test, test)
        self.relationship = relationship

        relationship_kwargs = relationship_kwargs or {}
        if alpha is not None:
            relationship_kwargs.update({"alpha": alpha})
        self.relationship_kwargs = relationship_kwargs

        if isinstance(samples, str):
            samples = [samples]
        elif samples is None:
            samples = []
        self.samples = samples
        super().__init__(
            check_fn=self.test,
            groupby=groupby,
            element_wise=False,
            name=name,
            error=error,
            raise_warning=raise_warning,
            n_failure_cases=n_failure_cases,
            title=title,
            description=description,
            statistics=statistics,
            strategy=strategy,
            **check_kwargs,
        )
