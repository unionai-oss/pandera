# pylint: disable=no-member,redefined-outer-name,unused-argument
# pylint: disable=unused-variable
"""Unit tests for pandera API extensions."""

from typing import Any, Optional, Union

import pandas as pd
import pytest

import pandera as pa
import pandera.strategies as st
from pandera import DataType, extensions
from pandera.checks import Check


def test_custom_checks_in_dir(extra_registered_checks):
    """Ensures that autocomplete works with registered custom checks."""
    assert "no_param_check" in dir(pa.Check)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([10, 10, 10]),
        pd.DataFrame([[10, 10, 10], [10, 10, 10]]),
    ],
)
def test_register_vectorized_custom_check(
    custom_check_teardown: None, data: Union[pd.Series, pd.DataFrame]
) -> None:
    """Test registering a vectorized custom check."""

    @extensions.register_check_method(
        statistics=["val"],
        supported_types=(pd.Series, pd.DataFrame),
        check_type="vectorized",
    )
    def custom_check(pandas_obj, *, val):
        return pandas_obj == val

    check = Check.custom_check(val=10)
    check_result = check(data)
    assert check_result.check_passed

    for kwargs in [
        {"element_wise": True},
        {"element_wise": False},
        {"groupby": "column"},
        {"groups": ["group1", "group2"]},
    ]:
        with pytest.warns(UserWarning):
            Check.custom_check(val=10, **kwargs)

    with pytest.raises(
        ValueError,
        match="method with name 'custom_check' already defined",
    ):
        # pylint: disable=function-redefined
        @extensions.register_check_method(statistics=["val"])
        def custom_check(pandas_obj, val):  # noqa
            return pandas_obj != val


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([10, 10, 10]),
        pd.DataFrame([[10, 10, 10], [10, 10, 10]]),
    ],
)
def test_register_element_wise_custom_check(
    custom_check_teardown: None, data: Union[pd.Series, pd.DataFrame]
) -> None:
    """Test registering an element-wise custom check."""

    @extensions.register_check_method(
        statistics=["val"],
        supported_types=(pd.Series, pd.DataFrame),
        check_type="element_wise",
    )
    def custom_check(element, *, val):
        return element == val

    check = Check.custom_check(val=10)
    check_result = check(data)
    assert check_result.check_passed

    for kwargs in [
        {"element_wise": True},
        {"element_wise": False},
        {"groupby": "column"},
        {"groups": ["group1", "group2"]},
    ]:
        with pytest.warns(UserWarning):
            Check.custom_check(val=10, **kwargs)

    with pytest.raises(
        ValueError,
        match="Element-wise checks should support DataFrame and Series "
        "validation",
    ):

        @extensions.register_check_method(
            supported_types=pd.Series,
            check_type="element_wise",
        )
        def invalid_custom_check(*args):
            pass


def test_register_custom_groupby_check(custom_check_teardown: None) -> None:
    """Test registering a custom groupby check."""

    @extensions.register_check_method(
        statistics=["group_a", "group_b"],
        supported_types=(pd.Series, pd.DataFrame),
        check_type="groupby",
    )
    def custom_check(dict_groups, *, group_a, group_b):
        """
        Test that the mean values in group A is larger than that of group B.

        Note that this function can handle groups of both dataframes and
        series.
        """
        return (
            dict_groups[group_a].values.mean()
            > dict_groups[group_b].values.mean()
        )

    # column groupby check
    data_column_check = pd.DataFrame(
        {
            "col1": [20, 20, 10, 10],
            "col2": list("aabb"),
        }
    )

    schema_column_check = pa.DataFrameSchema(
        {
            "col1": pa.Column(
                int,
                Check.custom_check(group_a="a", group_b="b", groupby="col2"),
            ),
            "col2": pa.Column(str),
        }
    )
    assert isinstance(schema_column_check(data_column_check), pd.DataFrame)

    # dataframe groupby check
    data_df_check = pd.DataFrame(
        {
            "col1": [20, 20, 10, 10],
            "col2": [30, 30, 5, 5],
            "col3": [10, 10, 1, 1],
        },
        index=pd.Index(list("aabb"), name="my_index"),
    )
    schema_df_check = pa.DataFrameSchema(
        columns={
            "col1": pa.Column(int),
            "col2": pa.Column(int),
            "col3": pa.Column(int),
        },
        index=pa.Index(str, name="my_index"),
        checks=Check.custom_check(
            group_a="a", group_b="b", groupby="my_index"
        ),
    )
    assert isinstance(schema_df_check(data_df_check), pd.DataFrame)

    for kwargs in [{"element_wise": True}, {"element_wise": False}]:
        with pytest.warns(UserWarning):
            Check.custom_check(val=10, **kwargs)


@pytest.mark.parametrize(
    "supported_types",
    [
        1,
        10.0,
        "foo",
        {"foo": "bar"},
        {1: 10},
        ["foo", "bar"],
        [1, 10],
        ("foo", "bar"),
        (1, 10),
    ],
)
def test_register_check_invalid_supported_types(supported_types: Any) -> None:
    """Test that TypeError is raised on invalid supported_types arg."""
    with pytest.raises(TypeError):

        @extensions.register_check_method(supported_types=supported_types)
        def custom_check(*args, **kwargs):
            pass


@pytest.mark.skipif(
    not st.HAS_HYPOTHESIS, reason='needs "strategies" module dependencies'
)
def test_register_check_with_strategy(custom_check_teardown: None) -> None:
    """Test registering a custom check with a data generation strategy."""

    import hypothesis  # pylint: disable=import-outside-toplevel,import-error

    def custom_ge_strategy(
        pandas_dtype: DataType,
        strategy: Optional[st.SearchStrategy] = None,
        *,
        min_value: Any,
    ) -> st.SearchStrategy:
        if strategy is None:
            return st.pandas_dtype_strategy(
                pandas_dtype,
                min_value=min_value,
                exclude_min=False,
            )
        return strategy.filter(lambda x: x > min_value)

    @extensions.register_check_method(
        statistics=["min_value"], strategy=custom_ge_strategy
    )
    def custom_ge_check(pandas_obj, *, min_value):
        return pandas_obj >= min_value

    check = Check.custom_ge_check(min_value=0)
    strat = check.strategy(pa.Int)
    with pytest.warns(hypothesis.errors.NonInteractiveExampleWarning):
        assert strat.example() >= 0


def test_schema_model_field_kwarg(custom_check_teardown: None) -> None:
    """Test that registered checks can be specified in a Field."""
    # pylint: disable=missing-class-docstring,too-few-public-methods

    @extensions.register_check_method(
        statistics=["val"],
        supported_types=(pd.Series, pd.DataFrame),
        check_type="vectorized",
    )
    def custom_gt(pandas_obj, val):
        return pandas_obj > val

    @extensions.register_check_method(
        statistics=["min_value", "max_value"],
        supported_types=(pd.Series, pd.DataFrame),
        check_type="vectorized",
    )
    def custom_in_range(pandas_obj, min_value, max_value):
        return (min_value <= pandas_obj) & (pandas_obj <= max_value)

    class Schema(pa.SchemaModel):
        """Schema that uses registered checks in Field."""

        col1: pa.typing.Series[int] = pa.Field(custom_gt=100)
        col2: pa.typing.Series[float] = pa.Field(
            custom_in_range={"min_value": -10, "max_value": 10}
        )

        class Config:
            coerce = True

    data = pd.DataFrame(
        {
            "col1": [101, 1000, 2000],
            "col2": [-5.0, 0.0, 6.0],
        }
    )
    Schema.validate(data)

    for invalid_data in [
        pd.DataFrame({"col1": [0], "col2": [-10.0]}),
        pd.DataFrame({"col1": [1000], "col2": [-100.0]}),
    ]:
        with pytest.raises(pa.errors.SchemaError):
            Schema.validate(invalid_data)


def test_register_before_schema_definitions() -> None:
    """Test that custom checks need to be registered before use."""
    # pylint: disable=missing-class-docstring,too-few-public-methods
    # pylint: disable=function-redefined

    with pytest.raises(
        pa.errors.SchemaInitError,
        match="custom check 'custom_eq' is not available",
    ):

        class Schema1(pa.SchemaModel):
            col: pa.typing.Series[int] = pa.Field(custom_eq=1)

    with pytest.raises(AttributeError):
        pa.Check.custom_eq(1)

    @extensions.register_check_method(statistics=["val"])
    def custom_eq(pandas_obj, val):
        return pandas_obj == val

    class Schema2(pa.SchemaModel):  # noqa F811
        col: pa.typing.Series[int] = pa.Field(custom_eq=1)

    pa.Check.custom_eq(1)
