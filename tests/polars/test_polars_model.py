"""Unit tests for Polars dataframe model."""

import sys
import warnings
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
from packaging import version
from polars.testing.parametric import column, dataframes

import pandera.backends.polars.utils as polars_utils
import pandera.engines.polars_engine as pe
from pandera.config import CONFIG
from pandera.errors import ParserError, SchemaError, SchemaErrors
from pandera.polars import (
    Column,
    DataFrameModel,
    DataFrameSchema,
    Field,
    PolarsData,
    check,
    dataframe_check,
)
from pandera.typing import FieldType


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


def test_field_type_presence_and_nullability():
    """Test the ``FieldType`` presence and nullability contract."""

    class Model(DataFrameModel):
        items: pl.List
        nullable_values: FieldType[int | None]
        optional_presence: FieldType[int] | None
        legacy_optional_presence: int | None
        explicit_nullable: int = Field(nullable=True)

    schema = Model.to_schema()
    assert schema.columns["items"].dtype == Column(pl.List).dtype
    assert schema.columns["nullable_values"].nullable
    assert schema.columns["nullable_values"].required
    assert not schema.columns["optional_presence"].required
    assert not schema.columns["optional_presence"].nullable
    # bare ``T | None`` keeps its historical optional-presence meaning
    assert not schema.columns["legacy_optional_presence"].required
    assert not schema.columns["legacy_optional_presence"].nullable
    assert schema.columns["explicit_nullable"].nullable
    assert Model.items == "items"
    assert isinstance(Model.optional_presence, str)


def test_field_type_contract():
    """The typing-only field marker composes with runtime field metadata."""

    class Model(DataFrameModel):
        checked: FieldType[
            int,
            Field(
                alias="renamed",
                description="checked field",
                metadata={"source": "typing-field"},
                title="Checked",
                unique=True,
                gt=0,
            ),
        ]
        nullable: FieldType[int | None, Field()]
        optional: FieldType[str] | None
        assigned: FieldType[int] = Field(description="assigned")

    schema = Model.to_schema()
    checked = schema.columns["renamed"]
    assert checked.required
    assert not checked.nullable
    assert checked.description == "checked field"
    assert checked.metadata == {"source": "typing-field"}
    assert checked.title == "Checked"
    assert checked.unique
    assert checked.checks
    assert schema.columns["nullable"].required
    assert schema.columns["nullable"].nullable
    assert not schema.columns["optional"].required
    assert schema.columns["assigned"].description == "assigned"
    assert Model.checked == "renamed"

    valid = pl.DataFrame({"renamed": [1], "nullable": [1], "assigned": [2]})
    Model.validate(valid)
    with pytest.raises(SchemaError):
        Model.validate(valid.with_columns(pl.lit(0).alias("renamed")))


def test_field_type_metadata_and_inheritance():
    """Test ``FieldType`` metadata and inheritance with Polars."""

    class Parent(DataFrameModel):
        inherited: FieldType[int, Field(description="inherited")]
        overridden: FieldType[int, Field(description="parent")]

    class Child(Parent):
        overridden: FieldType[int, Field(description="overridden")]
        annotated: FieldType[
            str, Field(alias="annotated_name", title="Annotated")
        ]
        optional: FieldType[int, Field(description="optional", required=False)]

    schema = Child.to_schema()
    assert schema.columns["inherited"].description == "inherited"
    assert schema.columns["overridden"].description == "overridden"
    assert schema.columns["annotated_name"].title == "Annotated"
    assert schema.columns["optional"].description == "optional"
    assert not schema.columns["optional"].required
    assert Child.inherited == "inherited"
    assert Child.annotated == "annotated_name"


@pytest.mark.parametrize(
    "column_mod,exception_cls",
    [
        # this modification will cause a InvalidOperationError since casting the
        # values in ldf_basic will cause the error outside of pandera validation
        pytest.param(
            {"string_col": pl.Int64},
            pl.exceptions.InvalidOperationError,
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


def test_field_type_metadata_propagation():
    """``FieldType[T, pa.Field(...)]`` should propagate field metadata
    (description, title, unique, checks, etc.) to
    the polars schema. See
    https://github.com/unionai-oss/pandera/issues/2110.
    """

    class Schema(DataFrameModel):
        name: FieldType[str, Field(description="Name of the person")]
        age: int = Field(ge=0, description="Age of the person")
        val: FieldType[float, Field(ge=0.0, description="A value")]
        identifier: FieldType[int, Field(unique=True, title="Identifier")]
        tag: FieldType[str, Field(metadata={"k": "v"})]

    schema = Schema.to_schema()

    assert schema.columns["name"].description == "Name of the person"
    assert schema.columns["age"].description == "Age of the person"
    assert schema.columns["val"].description == "A value"
    assert schema.columns["identifier"].unique is True
    assert schema.columns["identifier"].title == "Identifier"
    assert schema.columns["tag"].metadata == {"k": "v"}

    # ``ge`` check defined inside the FieldType metadata should also
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


def test_field_type_metadata_no_dedup():
    """Independent ``FieldType`` metadata objects must remain distinct."""

    class ModelA(DataFrameModel):
        value: FieldType[int, Field(ge=18, le=100)]

    class ModelB(DataFrameModel):
        value: FieldType[int, Field(unique=True, title="ID")]

    schema_a = ModelA.to_schema()
    schema_b = ModelB.to_schema()

    assert len(schema_a.columns["value"].checks) == 2
    assert schema_b.columns["value"].unique is True
    assert schema_b.columns["value"].title == "ID"
    # ModelB should not have inherited ModelA's range checks.
    assert schema_b.columns["value"].checks == []


@pytest.fixture
def simulate_polars_1_42_1(monkeypatch):
    """Simulate the concat deprecation introduced in polars>=1.42.1."""
    if polars_utils.polars_version().release >= (1, 42, 1):
        return

    monkeypatch.setattr(
        polars_utils,
        "polars_version",
        lambda: version.parse("1.42.1"),
    )

    original_concat = pl.concat

    def _simulate_polars_1_42_1_concat(*args, **kwargs):
        how = kwargs.get("how")
        if how == "horizontal":
            warnings.warn(
                "the default behavior of how='horizontal' for concat is "
                "deprecated and will require equal heights in the next "
                "breaking release. Use how='horizontal_extend' to keep "
                "the current behavior.",
                DeprecationWarning,
            )
        elif how == "horizontal_extend":
            kwargs = dict(kwargs)
            kwargs["how"] = "horizontal"
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(pl, "concat", _simulate_polars_1_42_1_concat)


def test_isin_check_lazy_validation_no_deprecation_warning(
    simulate_polars_1_42_1,
):
    """Issue #2409 regression test for lazy ``isin`` validation."""

    class Schema(DataFrameModel):
        string_col: str = Field(isin=[*"abc"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        valid_df = pl.DataFrame({"string_col": ["a", "b", "c"]})
        Schema.validate(valid_df, lazy=True)

        invalid_df = pl.DataFrame({"string_col": ["a", "b", "z"]})
        with pytest.raises(SchemaErrors) as exc_info:
            Schema.validate(invalid_df, lazy=True)

    message = str(exc_info.value)
    assert "CHECK_ERROR" not in message
    assert "DeprecationWarning" not in message
    assert "isin" in message
    assert exc_info.value.failure_cases["failure_case"].to_list() == ["z"]


def test_nullable_check_lazy_validation_no_deprecation_warning(
    simulate_polars_1_42_1,
):
    """Issue #2409 regression test for lazy nullable validation."""

    class Schema(DataFrameModel):
        col: str = Field(nullable=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(SchemaErrors) as exc_info:
            Schema.validate(pl.DataFrame({"col": ["a", None, "c"]}), lazy=True)

    message = str(exc_info.value)
    assert "DeprecationWarning" not in message
    assert "SERIES_CONTAINS_NULLS" in message


def test_unique_check_lazy_validation_no_deprecation_warning(
    simulate_polars_1_42_1,
):
    """Issue #2409 regression test for lazy uniqueness validation."""

    class Schema(DataFrameModel):
        col: str = Field(unique=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(SchemaErrors) as exc_info:
            Schema.validate(pl.DataFrame({"col": ["a", "a", "c"]}), lazy=True)

    message = str(exc_info.value)
    assert "DeprecationWarning" not in message
    assert "SERIES_CONTAINS_DUPLICATES" in message


def test_coercion_failure_no_deprecation_warning(simulate_polars_1_42_1):
    """Issue #2409 regression test for generic coercion failures."""

    class Schema(DataFrameModel):
        int_col: int = Field(coerce=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        bad_df = pl.DataFrame({"int_col": ["a", "b", "not-a-number"]})
        with pytest.raises(SchemaErrors) as exc_info:
            Schema.validate(bad_df, lazy=True)

    message = str(exc_info.value)
    assert "DeprecationWarning" not in message
    assert "DATATYPE_COERCION" in message or "WRONG_DATATYPE" in message


def test_category_coercion_failure_no_deprecation_warning(
    simulate_polars_1_42_1,
):
    """Issue #2409 regression test for category coercion failures."""
    cat_dtype = pe.Category(categories=["a", "b", "c"])
    lazyframe = pl.DataFrame({"col": ["a", "b", "not-a-category"]}).lazy()
    data_container = PolarsData(lazyframe, key="col")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ParserError) as exc_info:
            cat_dtype.try_coerce(data_container)

    message = str(exc_info.value)
    assert "DeprecationWarning" not in message
    assert "Invalid categories" in message
