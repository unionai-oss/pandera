"""Unit tests for Polars dataframe model."""

import sys
from datetime import datetime
from typing import Optional

try:  # python 3.9+
    from typing import Annotated  # type: ignore
except ImportError:
    from typing import Annotated  # type: ignore

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing.parametric import column, dataframes

import pandera.engines.polars_engine as pe
from pandera.errors import SchemaError
from pandera.polars import (
    Column,
    DataFrameModel,
    DataFrameSchema,
    Field,
    PolarsData,
    check,
    dataframe_check,
)

try:
    import narwhals  # noqa: F401

    narwhals_installed = True
except ImportError:
    narwhals_installed = False


@pytest.fixture
def ldf_model_basic():
    class BasicModel(DataFrameModel):
        string_col: str
        int_col: int

    return BasicModel


@pytest.fixture
def ldf_schema_basic():
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        }
    )


@pytest.fixture
def ldf_model_with_fields():
    class ModelWithFields(DataFrameModel):
        string_col: str = Field(isin=[*"abc"])
        int_col: int = Field(ge=0)

    return ModelWithFields


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="coerce_dtype not implemented in Narwhals backend (used by DataFrameModel.empty())",
    strict=True,
)
def test_empty() -> None:
    """Test to generate an empty DataFrameModel."""

    class Schema(DataFrameModel):
        a: float
        b: int
        c: str
        d: datetime

    df = Schema.empty()
    assert df.is_empty()
    assert Schema.validate(df).is_empty()  # type: ignore [attr-defined]


def test_empty_no_columns() -> None:
    """Test empty() on a DataFrameModel with no field annotations."""

    class EmptySchema(DataFrameModel):
        pass

    df = EmptySchema.empty()
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (0, 0)


@pytest.fixture
def ldf_model_with_custom_column_checks():
    class ModelWithCustomColumnChecks(DataFrameModel):
        string_col: str
        int_col: int

        @check("string_col")
        @classmethod
        def custom_isin(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(pl.col(data.key).is_in([*"abc"]))

        @check("int_col")
        @classmethod
        def custom_ge(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(pl.col(data.key).ge(0))

    return ModelWithCustomColumnChecks


@pytest.fixture
def ldf_model_with_custom_dataframe_checks():
    class ModelWithCustomDataFrameChecks(DataFrameModel):
        string_col: str
        int_col: int

        @dataframe_check
        @classmethod
        def not_empty(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(pl.len().alias("len").gt(0))

    return ModelWithCustomDataFrameChecks


@pytest.fixture
def ldf_basic():
    """Basic polars lazy dataframe fixture."""
    return pl.DataFrame(
        {
            "string_col": ["a", "b", "c"],
            "int_col": [0, 1, 2],
        }
    ).lazy()


def test_model_schema_equivalency(
    ldf_model_basic: DataFrameModel,
    ldf_schema_basic: DataFrameSchema,
):
    """Test that polars DataFrameModel and DataFrameSchema are equivalent."""
    ldf_schema_basic.name = "BasicModel"
    assert ldf_model_basic.to_schema() == ldf_schema_basic


def test_model_schema_equivalency_with_optional():
    class ModelWithOptional(DataFrameModel):
        string_col: str | None
        int_col: int

    schema = DataFrameSchema(
        name="ModelWithOptional",
        columns={
            "string_col": Column(pl.Utf8, required=False),
            "int_col": Column(pl.Int64),
        },
    )
    assert ModelWithOptional.to_schema() == schema


ErrorCls = (
    pl.exceptions.InvalidOperationError
    if pe.polars_version().release >= (1, 0, 0)
    else pl.exceptions.ComputeError
)


@pytest.mark.parametrize(
    "column_mod,exception_cls",
    [
        # this modification will cause a InvalidOperationError since casting the
        # values in ldf_basic will cause the error outside of pandera validation
        pytest.param(
            {"string_col": pl.Int64},
            ErrorCls,
            marks=pytest.mark.xfail(
                condition=narwhals_installed,
                reason="Narwhals raises narwhals.exceptions.InvalidOperationError, not polars.exceptions.InvalidOperationError",
                strict=True,
            ),
        ),
        # this modification will cause a SchemaError since schema validation
        # can actually catch the type mismatch
        ({"int_col": pl.Utf8}, SchemaError),
        ({"int_col": pl.Float64}, SchemaError),
    ],
)
def test_basic_model(
    column_mod,
    exception_cls,
    ldf_model_basic: DataFrameModel,
    ldf_basic: pl.LazyFrame,
):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.pipe(ldf_model_basic.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_df = ldf_basic.cast(column_mod)

    with pytest.raises(exception_cls):
        invalid_df.pipe(ldf_model_basic.validate).collect()


def test_model_with_fields(ldf_model_with_fields, ldf_basic):
    query = ldf_basic.pipe(ldf_model_with_fields.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_df = ldf_basic.with_columns(
        string_col=pl.lit("x"), int_col=pl.lit(-1)
    )
    with pytest.raises(SchemaError):
        invalid_df.pipe(ldf_model_with_fields.validate).collect()


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Polars-style custom check functions incompatible with Narwhals backend",
    strict=True,
)
def test_model_with_custom_column_checks(
    ldf_model_with_custom_column_checks,
    ldf_basic,
):
    query = ldf_basic.pipe(ldf_model_with_custom_column_checks.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    invalid_df = ldf_basic.with_columns(
        string_col=pl.lit("x"), int_col=pl.lit(-1)
    )
    with pytest.raises(SchemaError):
        invalid_df.pipe(ldf_model_with_custom_column_checks.validate).collect()


@pytest.mark.xfail(
    condition=narwhals_installed,
    reason="Polars-style custom check functions incompatible with Narwhals backend",
    strict=True,
)
def test_model_with_custom_dataframe_checks(
    ldf_model_with_custom_dataframe_checks,
    ldf_basic,
):
    query = ldf_basic.pipe(ldf_model_with_custom_dataframe_checks.validate)
    df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)

    # remove all rows
    invalid_df = ldf_basic.filter(pl.lit(False))
    with pytest.raises(SchemaError):
        invalid_df.pipe(
            ldf_model_with_custom_dataframe_checks.validate
        ).collect()


@pytest.fixture
def schema_with_list_type():
    return DataFrameSchema(
        name="ModelWithNestedDtypes",
        columns={
            "list_col": Column(pl.List(pl.Utf8)),
        },
    )


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="standard collection generics are not supported in python < 3.9",
)
def test_polars_python_list_df_model(schema_with_list_type):
    class ModelWithNestedDtypes(DataFrameModel):
        # pylint: disable=unsubscriptable-object
        list_col: list[str]

    schema = ModelWithNestedDtypes.to_schema()
    assert schema_with_list_type == schema


@pytest.mark.parametrize(
    "time_zone",
    [
        None,
        pytest.param(
            "UTC",
            marks=pytest.mark.xfail(
                condition=narwhals_installed,
                reason="Narwhals engine dtype comparison fails for tz-aware polars Datetime",
                strict=True,
            ),
        ),
        pytest.param(
            "GMT",
            marks=pytest.mark.xfail(
                condition=narwhals_installed,
                reason="Narwhals engine dtype comparison fails for tz-aware polars Datetime",
                strict=True,
            ),
        ),
        pytest.param(
            "EST",
            marks=pytest.mark.xfail(
                condition=narwhals_installed,
                reason="Narwhals engine dtype comparison fails for tz-aware polars Datetime",
                strict=True,
            ),
        ),
    ],
)
@given(st.data())
def test_dataframe_schema_with_tz_agnostic_dates(time_zone, data):
    strategy = dataframes(
        column("datetime_col", dtype=pl.Datetime()),
        lazy=True,
        min_size=10,
        max_size=10,
        allow_null=False,
    )
    lf = data.draw(strategy)
    lf = lf.cast({"datetime_col": pl.Datetime(time_zone=time_zone)})

    class ModelTZAgnosticKwargs(DataFrameModel):
        datetime_col: pe.DateTime = Field(
            dtype_kwargs={"time_zone_agnostic": True}
        )

    class ModelTZSensitiveKwargs(DataFrameModel):
        datetime_col: pe.DateTime = Field(
            dtype_kwargs={"time_zone_agnostic": False}
        )

    class ModelTZAgnosticAnnotated(DataFrameModel):
        datetime_col: Annotated[pe.DateTime, True, "us", None]

    class ModelTZSensitiveAnnotated(DataFrameModel):
        datetime_col: Annotated[pe.DateTime, False, "us", None]

    for tz_agnostic_model in (
        ModelTZAgnosticKwargs,
        ModelTZAgnosticAnnotated,
    ):
        tz_agnostic_model.validate(lf)

    for tz_sensitive_model in (
        ModelTZSensitiveKwargs,
        ModelTZSensitiveAnnotated,
    ):
        if time_zone:
            with pytest.raises(SchemaError):
                tz_sensitive_model.validate(lf)
