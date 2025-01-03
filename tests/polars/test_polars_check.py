"""Unit tests for polars check class."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.api.polars.utils import get_lazyframe_schema
from pandera.constants import CHECK_OUTPUT_KEY


@pytest.fixture
def column_lf():
    return pl.LazyFrame({"col": pl.Series([1, 2, 3, 4], dtype=int)})


@pytest.fixture
def lf():
    return pl.LazyFrame(
        {
            "col_1": pl.Series([1, 2, 3, 4], dtype=int),
            "col_2": pl.Series([1, 2, 3, 4], dtype=int),
        }
    )


def _column_check_fn_df_out(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key).ge(0))


def _column_check_fn_scalar_out(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key).ge(0).all())


@pytest.mark.parametrize(
    "check_fn, invalid_data, expected_output, ignore_na",
    [
        [
            _column_check_fn_df_out,
            [-1, 2, 3, -2],
            [False, True, True, False],
            False,
        ],
        [_column_check_fn_scalar_out, [-1, 2, 3, -2], [False], False],
        [
            _column_check_fn_df_out,
            [-1, 2, 3, None],
            [False, True, True, True],
            True,
        ],
        [_column_check_fn_scalar_out, [-1, 2, 3, None], [False], True],
    ],
)
def test_polars_column_check(
    column_lf,
    check_fn,
    invalid_data,
    expected_output,
    ignore_na,
):
    check = pa.Check(check_fn, ignore_na=ignore_na)
    check_result = check(column_lf, column="col")
    assert check_result.check_passed.collect().item()

    invalid_lf = column_lf.with_columns(col=pl.Series(invalid_data, dtype=int))
    invalid_check_result = check(invalid_lf, column="col")
    assert not invalid_check_result.check_passed.collect().item()
    assert (
        invalid_check_result.check_output.collect()[CHECK_OUTPUT_KEY].to_list()
        == expected_output
    )


def _df_check_fn_df_out(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col("*").ge(0))


def _df_check_fn_col_out(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col("col_1").ge(pl.col("col_2")))


def _df_check_fn_scalar_out(data: pa.PolarsData):
    return data.lazyframe.select(pl.col("*").ge(0).all()).select(
        pl.all_horizontal("*")
    )


@pytest.mark.parametrize(
    "check_fn, invalid_data, expected_output",
    [
        [
            _df_check_fn_df_out,
            {
                "col_1": pl.Series([-1, 2, -3, 4]),
                "col_2": pl.Series([1, 2, 3, -4]),
            },
            [False, True, False, False],
        ],
        [
            _df_check_fn_col_out,
            {
                "col_1": pl.Series([1, 2, 3, 4]),
                "col_2": pl.Series([2, 1, 2, 5]),
            },
            [False, True, True, False],
        ],
        [
            _df_check_fn_scalar_out,
            {
                "col_1": pl.Series([-1, 2, 3, 4]),
                "col_2": pl.Series([2, 1, 2, 5]),
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
    check_result = check(lf, column=r"^col_\d+$")
    assert check_result.check_passed.collect().item()

    invalid_lf = lf.with_columns(**invalid_data)
    invalid_check_result = check(invalid_lf)
    assert not invalid_check_result.check_passed.collect().item()
    assert (
        invalid_check_result.check_output.collect()[CHECK_OUTPUT_KEY].to_list()
        == expected_output
    )


def _element_wise_check_fn(x):
    return x > 0


def test_polars_element_wise_column_check(column_lf):

    check = pa.Check(_element_wise_check_fn, element_wise=True)
    col_schema = pa.Column(int, name="col", checks=check)
    validated_data = col_schema.validate(column_lf)
    assert validated_data.collect().equals(column_lf.collect())

    invalid_lf = column_lf.with_columns(
        col=pl.Series([-1, 2, 3, -2], dtype=int)
    )
    try:
        col_schema.validate(invalid_lf)
    except pa.errors.SchemaError as exc:
        exc.failure_cases.equals(pl.DataFrame({"col": [-1, -2]}))


def test_polars_element_wise_dataframe_check(lf):

    check = pa.Check(_element_wise_check_fn, element_wise=True)
    schema = pa.DataFrameSchema(dtype=int, checks=check)
    validated_data = schema.validate(lf)
    assert validated_data.collect().equals(lf.collect())

    for col in get_lazyframe_schema(lf):
        invalid_lf = lf.with_columns(**{col: pl.Series([-1, 2, -4, 3])})
        try:
            schema.validate(invalid_lf)
        except pa.errors.SchemaError as exc:
            exc.failure_cases.equals(pl.DataFrame({col: [-1, -4]}))


def test_polars_element_wise_dataframe_different_dtypes(column_lf):

    # Custom check function
    def check_gt_2(v: int) -> bool:
        return v > 2

    def check_len_ge_2(v: str) -> bool:
        return len(v) >= 2

    lf = column_lf.with_columns(
        str_col=pl.Series(["aaa", "bb", "c", "dd"], dtype=str)
    )

    schema = pa.DataFrameSchema(
        {
            "col": pa.Column(
                dtype=int, checks=pa.Check(check_gt_2, element_wise=True)
            ),
            "str_col": pa.Column(
                dtype=str, checks=pa.Check(check_len_ge_2, element_wise=True)
            ),
        }
    )

    try:
        schema.validate(lf, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases["failure_case"].to_list() == ["1", "2", "c"]


def test_polars_custom_check():
    """Test that custom checks with more complex expressions are supported."""

    lf = pl.LazyFrame(
        {"column1": [None, "x", "y"], "column2": ["a", None, "c"]}
    )

    def custom_check(data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.when(
                pl.col("column1").is_null(),
                pl.col(data.key).is_null(),
            )
            .then(False)
            .otherwise(True)
        )

    schema = pa.DataFrameSchema(
        {
            "column1": pa.Column(str, nullable=True),
            "column2": pa.Column(
                str, nullable=True, checks=pa.Check(custom_check)
            ),
        }
    )

    validated_lf = schema.validate(lf)
    assert isinstance(validated_lf, pl.LazyFrame)

    invalid_lf = pl.LazyFrame(
        {"column1": [None, "x", "y"], "column2": [None, None, "c"]}
    )

    with pytest.raises(pa.errors.SchemaError):
        schema.validate(invalid_lf)
