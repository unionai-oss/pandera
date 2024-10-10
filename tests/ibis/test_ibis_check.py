"""Unit tests for the Ibis check backend."""

import pytest

import pandas as pd
import ibis
import ibis.expr.types as ir
import pandera.ibis as pa
from pandera.backends.ibis.register import register_ibis_backends


@pytest.fixture(autouse=True)
def _register_ibis_backends():
    register_ibis_backends()


@pytest.fixture
def column_t():
    return ibis.memtable(pd.DataFrame({"col": [1, 2, 3, 4]}))


def _column_check_fn_col_out(data: pa.IbisData) -> ir.Table:
    return data.table[data.key] > 0


def _column_check_fn_scalar_out(data: pa.IbisData) -> ir.Table:
    return (data.table[data.key] > 0).all()


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


def _df_check_fn_df_out(data: pa.IbisData) -> ir.Table:
    return data.table.mutate(
        *[data.table[col] > 0 for col in data.table.columns]
    ).drop(data.table.columns)


def _df_check_fn_col_out(data: pa.IbisData) -> ir.logical.BooleanColumn:
    return data.table[data.key] > 0


def _df_check_fn_scalar_out(data: pa.IbisData) -> ir.logical.BooleanScalar:
    acc = None
    for col in data.table.columns:
        _result = data.table[col] > 0
        acc = _result if acc is None else acc & _result

    return acc.all()


@pytest.mark.parametrize(
    "check_fn, invalid_data, expected_output",
    [
        [
            _df_check_fn_df_out,
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
            [False],
        ],
    ],
)
def test_polars_dataframe_check(
    lf,
    check_fn,
    invalid_data,
    expected_output,
):
    check = pa.Check(check_fn)
    raise NotImplementedError
    check_result = check(lf, column=r"^col_\d+$")
    assert check_result.check_passed.collect().item()

    invalid_lf = lf.with_columns(**invalid_data)
    invalid_check_result = check(invalid_lf)
    assert not invalid_check_result.check_passed.collect().item()
    assert (
        invalid_check_result.check_output.collect()[CHECK_OUTPUT_KEY].to_list()
        == expected_output
    )


def test_ibis_element_wise_column_check(): ...


def test_ibis_element_wise_dataframe_check(): ...


def test_ibis_element_wise_dataframe_different_dtypes(): ...


def test_ibis_custom_check(): ...
