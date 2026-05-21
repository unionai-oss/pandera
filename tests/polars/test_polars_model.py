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
from pandera.config import CONFIG
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
    condition=CONFIG.use_narwhals_backend,
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
                condition=CONFIG.use_narwhals_backend,
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
    condition=CONFIG.use_narwhals_backend,
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
    condition=CONFIG.use_narwhals_backend,
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
                condition=CONFIG.use_narwhals_backend,
                reason="Narwhals engine dtype comparison fails for tz-aware polars Datetime",
                strict=True,
            ),
        ),
        pytest.param(
            "GMT",
            marks=pytest.mark.xfail(
                condition=CONFIG.use_narwhals_backend,
                reason="Narwhals engine dtype comparison fails for tz-aware polars Datetime",
                strict=True,
            ),
        ),
        pytest.param(
            "EST",
            marks=pytest.mark.xfail(
                condition=CONFIG.use_narwhals_backend,
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


def test_model_field_access_returns_string():
    """Test that accessing DataFrameModel fields returns column names as strings.

    Regression test for issue #2297.
    """
    from pandera.typing.polars import Series

    class ModelWithSeries(DataFrameModel):
        a: Series[int]
        b: Series[float]

    class ModelWithBareTypes(DataFrameModel):
        x: int
        y: float

    # Both Series and bare type annotations should return strings
    assert isinstance(ModelWithSeries.a, str)
    assert isinstance(ModelWithSeries.b, str)
    assert isinstance(ModelWithBareTypes.x, str)
    assert isinstance(ModelWithBareTypes.y, str)

    # Verify the actual column names
    assert ModelWithSeries.a == "a"
    assert ModelWithSeries.b == "b"
    assert ModelWithBareTypes.x == "x"
    assert ModelWithBareTypes.y == "y"


def test_annotated_field_metadata_propagation():
    """``Annotated[T, pa.Field(...)]`` should propagate the embedded
    ``FieldInfo`` metadata (description, title, unique, checks, etc.) to
    the polars schema. See
    https://github.com/unionai-oss/pandera/issues/2110.
    """

    class Schema(DataFrameModel):
        name: Annotated[str, Field(description="Name of the person")]
        age: int = Field(ge=0, description="Age of the person")
        val: Annotated[float, Field(ge=0.0, description="A value")]
        identifier: Annotated[int, Field(unique=True, title="Identifier")]
        tag: Annotated[str, Field(metadata={"k": "v"})]

    schema = Schema.to_schema()

    assert schema.columns["name"].description == "Name of the person"
    assert schema.columns["age"].description == "Age of the person"
    assert schema.columns["val"].description == "A value"
    assert schema.columns["identifier"].unique is True
    assert schema.columns["identifier"].title == "Identifier"
    assert schema.columns["tag"].metadata == {"k": "v"}

    # ``ge`` check defined inside the Annotated FieldInfo should also
    # be applied during validation.
    valid = pl.DataFrame(
        {
            "name": ["Alice"],
            "age": [25],
            "val": [1.0],
            "identifier": [1],
            "tag": ["x"],
        }
    )
    Schema.validate(valid)

    invalid = valid.with_columns(pl.lit(-1.0).alias("val"))
    with pytest.raises(SchemaError):
        Schema.validate(invalid)


def test_annotated_field_no_metadata_dedup():
    """Two ``Annotated`` annotations using independent ``Field(...)``
    calls must not be deduplicated by Python's ``typing.Annotated`` cache.
    Without unique hashing on un-named ``FieldInfo`` instances, the second
    model would inadvertently inherit the first model's field configuration.
    """

    class ModelA(DataFrameModel):
        value: Annotated[int, Field(ge=18, le=100)]

    class ModelB(DataFrameModel):
        value: Annotated[int, Field(unique=True, title="ID")]

    schema_a = ModelA.to_schema()
    schema_b = ModelB.to_schema()

    assert len(schema_a.columns["value"].checks) == 2
    assert schema_b.columns["value"].unique is True
    assert schema_b.columns["value"].title == "ID"
    # ModelB should not have inherited ModelA's range checks.
    assert schema_b.columns["value"].checks == []
