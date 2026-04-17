"""Unit tests for the Ibis check backend."""

from contextlib import nullcontext

import ibis
import ibis.expr.types as ir
import pandas as pd
import pytest
from ibis import _
from ibis import selectors as s

import pandera.ibis as pa
from pandera.backends.ibis.register import register_ibis_backends
from pandera.constants import CHECK_OUTPUT_KEY

try:
    import narwhals  # noqa: F401

    narwhals_installed = True
except ImportError:
    narwhals_installed = False


@pytest.fixture(autouse=True, scope="module")
def _register_ibis_backends():
    register_ibis_backends()


@pytest.fixture
def column_t():
    return ibis.memtable(pd.DataFrame({"col": [1, 2, 3, 4]}))


@pytest.fixture
def t():
    return ibis.memtable(
        pd.DataFrame({"col_1": [1, 2, 3, 4], "col_2": [1, 2, 3, 4]})
    )


def _column_check_fn_col_out(data: pa.IbisData) -> ir.Table:
    return data.table[data.key] > 0


def _column_check_fn_scalar_out(data: pa.IbisData) -> ir.Table:
    return (data.table[data.key] > 0).all()


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Ibis-style check functions incompatible with Narwhals backend",
    strict=True,
)
@pytest.mark.parametrize(
    "check_fn, invalid_data, expected_output",
    [
        [_column_check_fn_col_out, [-1, 2, 3, -2], [False, True, True, False]],
        [_column_check_fn_scalar_out, [-1, 2, 3, -2], False],
    ],
)
def test_ibis_column_check(
    column_t,
    check_fn,
    invalid_data,
    expected_output,
):
    check = pa.Check(check_fn)
    check_result = check(column_t, column="col")
    assert check_result.check_passed.execute()

    invalid_df = ibis.memtable(pd.DataFrame({"col": invalid_data}))
    invalid_check_result = check(invalid_df, column="col")
    assert not invalid_check_result.check_passed.execute()

    check_output = invalid_check_result.check_output.execute()
    if isinstance(expected_output, list):
        assert check_output.tolist() == expected_output
    else:
        assert check_output == expected_output


def _df_check_fn_table_out(data: pa.IbisData) -> ir.Table:
    return data.table.select(s.across(s.numeric(), _ >= 0))


def _df_check_fn_dict_out(data: pa.IbisData) -> dict[str, ir.BooleanColumn]:
    return {col: data.table[col] >= 0 for col in data.table.columns}


def _df_check_fn_col_out(data: pa.IbisData) -> ir.BooleanColumn:
    return data.table["col_1"] >= data.table["col_2"]


def _df_check_fn_scalar_out(data: pa.IbisData) -> ir.BooleanScalar:
    acc = data.table[data.table.columns[0]] >= 0
    for col in data.table.columns[1:]:
        acc &= data.table[col] >= 0
    return acc.all()


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Ibis-style check functions incompatible with Narwhals backend",
    strict=True,
)
@pytest.mark.parametrize(
    "check_fn, invalid_data, expected_output",
    [
        [
            _df_check_fn_table_out,
            {
                "col_1": [-1, 2, -3, 4],
                "col_2": [1, 2, 3, -4],
            },
            [False, True, False, False],
        ],
        [
            _df_check_fn_dict_out,
            {
                "col_1": [-1, 2, -3, 4],
                "col_2": [1, 2, 3, -4],
            },
            [False, True, False, False],
        ],
        [
            _df_check_fn_col_out,
            {
                "col_1": [1, 2, 3, 4],
                "col_2": [2, 1, 2, 5],
            },
            [False, True, True, False],
        ],
        [
            _df_check_fn_scalar_out,
            {
                "col_1": [-1, 2, 3, 4],
                "col_2": [2, 1, 2, 5],
            },
            False,
        ],
    ],
)
def test_ibis_table_check(
    t,
    check_fn,
    invalid_data,
    expected_output,
):
    check = pa.Check(check_fn)
    check_result = check(t)
    assert check_result.check_passed.as_scalar().execute()

    invalid_t = ibis.memtable(pd.DataFrame(invalid_data))
    invalid_check_result = check(invalid_t)
    assert not invalid_check_result.check_passed.as_scalar().execute()

    if isinstance(invalid_check_result.check_output, ir.Table):
        output = (
            invalid_check_result.check_output[CHECK_OUTPUT_KEY]
            .to_pandas()
            .to_list()
        )
    elif isinstance(invalid_check_result.check_output, ir.BooleanColumn):
        output = invalid_check_result.check_output.to_pandas().to_list()
    elif isinstance(invalid_check_result.check_output, ir.BooleanScalar):
        output = invalid_check_result.check_output.as_scalar().execute()
    else:
        raise ValueError(
            f"Invalid check output type: {type(invalid_check_result.check_output)}"
        )

    assert output == expected_output


def _element_wise_check_fn(x: int) -> bool:
    return x > 0


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Element-wise checks incompatible with Narwhals backend",
    strict=True,
)
def test_ibis_element_wise_column_check(column_t):
    check = pa.Check(_element_wise_check_fn, element_wise=True)
    check_result = check(column_t, column="col")
    assert check_result.check_passed.execute()

    invalid_t = ibis.memtable(pd.DataFrame({"col": [-1, 2, 3, -2]}))
    invalid_check_result = check(invalid_t, column="col")
    assert not invalid_check_result.check_passed.execute()
    failure_cases = invalid_check_result.failure_cases.execute()
    expected_failure_cases = pd.DataFrame({"col": [-1, -2]})
    assert failure_cases.equals(expected_failure_cases)


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Element-wise checks incompatible with Narwhals backend",
    strict=True,
)
def test_ibis_element_wise_dataframe_check(t):
    check = pa.Check(_element_wise_check_fn, element_wise=True)
    check_result = check(t)
    assert check_result.check_passed.execute()

    invalid_t = ibis.memtable(
        pd.DataFrame({"col_1": [-1, 2, -3, 4], "col_2": [1, 2, 3, -4]})
    )
    invalid_check_result = check(invalid_t)
    assert not invalid_check_result.check_passed.execute()
    failure_cases = invalid_check_result.failure_cases.execute()
    expected_failure_cases = pd.DataFrame(
        {"col_1": [-1, -3, 4], "col_2": [1, 3, -4]}
    )
    assert failure_cases.equals(expected_failure_cases)


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Element-wise checks incompatible with Narwhals backend",
    strict=True,
)
def test_ibis_element_wise_dataframe_different_dtypes():
    # Custom check function
    def check_gt_2(v: int) -> bool:
        return v > 2

    def check_len_ge_2(v: str) -> bool:
        return len(v) >= 2

    t = ibis.memtable(
        pd.DataFrame(
            {"int_col": [1, 2, 3, 4], "str_col": ["aaa", "bb", "c", "dd"]}
        )
    )

    _check_gt_2 = pa.Check(check_gt_2, element_wise=True)
    _check_len_ge_2 = pa.Check(check_len_ge_2, element_wise=True)

    check_result = _check_gt_2(t, column="int_col")
    check_result.check_passed.execute()

    assert not check_result.check_passed.execute()
    assert check_result.failure_cases.to_pandas().equals(
        pd.DataFrame({"int_col": [1, 2]})
    )

    check_result = _check_len_ge_2(t, column="str_col")
    assert not check_result.check_passed.execute()
    assert check_result.failure_cases.to_pandas().equals(
        pd.DataFrame({"str_col": ["c"]})
    )


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Ibis-style custom check functions incompatible with Narwhals backend",
    strict=True,
)
def test_ibis_custom_check():
    t = ibis.memtable(
        pd.DataFrame(
            {"column1": [None, "x", "y"], "column2": ["a", None, "c"]}
        )
    )

    def custom_check(data: pa.IbisData) -> ir.Table:
        both_null = data.table.column1.isnull() & data.table.column2.isnull()
        return ~both_null

    check = pa.Check(custom_check)
    check_result = check(t)
    assert check_result.check_passed.execute()
    assert check_result.failure_cases.execute().empty

    invalid_t = ibis.memtable(
        pd.DataFrame(
            {"column1": [None, "x", "y"], "column2": [None, None, "c"]}
        )
    )
    invalid_check_result = check(invalid_t)
    assert not invalid_check_result.check_passed.execute()
    failure_cases = invalid_check_result.failure_cases.execute()
    expected_failure_cases = pd.DataFrame(
        {"column1": [None], "column2": [None]}
    )
    assert failure_cases.equals(expected_failure_cases)


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Custom check signature incompatible with Narwhals backend",
    strict=True,
)
def test_ibis_column_check_n_failure_cases(column_t):
    n_failure_cases = 2
    check = pa.Check(
        lambda data: data.table.select(s.across(s.numeric(), _ < 0)),
        n_failure_cases=n_failure_cases,
    )
    schema = pa.DataFrameSchema({"col": pa.Column(checks=check)})
    try:
        schema.validate(column_t, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases.shape[0] == n_failure_cases


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Custom check signature incompatible with Narwhals backend",
    strict=True,
)
def test_ibis_dataframe_check_n_failure_cases(t):
    n_failure_cases = 2
    check = pa.Check(
        lambda data: data.table.select(s.across(s.numeric(), _ < 0)),
        n_failure_cases=n_failure_cases,
    )
    schema = pa.DataFrameSchema(checks=check)
    try:
        schema.validate(t, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases.shape[0] == n_failure_cases


@pytest.mark.parametrize(
    "values,expectation",
    [
        pytest.param([None, None], nullcontext(), id="all_nulls_pass"),
        pytest.param([42, None], nullcontext(), id="mixed_nulls_pass"),
        pytest.param([-5, None], pytest.raises(pa.errors.SchemaError), id="invalid_value_fails"),
        pytest.param([10, 20], nullcontext(), id="all_valid_pass"),
    ],
)
def test_nullable_column_check(values, expectation):
    """Regression test for https://github.com/unionai-oss/pandera/issues/2294."""
    df = ibis.memtable({"my_value": values}).cast({"my_value": "float64"})
    schema = pa.DataFrameSchema(
        {
            "my_value": pa.Column(
                float,
                nullable=True,
                checks=[pa.Check.greater_than_or_equal_to(0)],
            )
        }
    )
    with expectation:
        assert isinstance(schema.validate(df), ibis.Table)
