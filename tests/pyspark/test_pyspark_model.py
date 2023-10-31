"""Unit tests for DataFrameModel module."""
# pylint:disable=abstract-method

from contextlib import nullcontext as does_not_raise
from typing import Optional
from pyspark.sql import DataFrame
import pyspark.sql.types as T
import pytest

import pandera
import pandera.api.extensions as pax
import pandera.pyspark as pa
from pandera.config import PanderaConfig, ValidationDepth
from pandera.pyspark import DataFrameModel, DataFrameSchema, Field
from tests.pyspark.conftest import spark_df
from pandera.api.pyspark.model import docstring_substitution
from pandera.errors import (
    SchemaDefinitionError,
)


def test_schema_with_bare_types():
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        """Test class"""

        a: int
        b: str
        c: float

    expected = pa.DataFrameSchema(
        name="Model",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(str),
            "c": pa.Column(float),
        },
        # The Dataframe Model uses class doc as description if not explicitly defined in config class
        description="Test class",
    )

    assert expected == Model.to_schema()


def test_schema_with_bare_types_and_field():
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        """Model Schema"""

        a: int = Field()
        b: str = Field()
        c: float = Field()

    expected = DataFrameSchema(
        name="Model",
        columns={
            "a": pa.Column(int),
            "b": pa.Column(str),
            "c": pa.Column(float),
        },
        description="Model Schema",
    )

    assert expected == Model.to_schema()


def test_schema_with_bare_types_field_and_checks(spark):
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        """Model Schema"""

        a: str = Field(str_startswith="B")
        b: int = Field(gt=6)
        c: float = Field()

    expected = DataFrameSchema(
        name="Model",
        columns={
            "a": pa.Column(str, checks=pa.Check.str_startswith("B")),
            "b": pa.Column(int, checks=pa.Check.gt(6)),
            "c": pa.Column(float),
        },
        description="Model Schema",
    )

    assert expected == Model.to_schema()

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99.99)]

    spark_schema = T.StructType(
        [
            T.StructField("a", T.StringType(), False),  # should fail
            T.StructField("b", T.IntegerType(), False),  # should fail
            T.StructField("c", T.FloatType(), False),
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = Model.validate(check_obj=df_fail)
    assert df_out.pandera.errors is not None


def test_schema_with_bare_types_field_type(spark):
    """
    Test that DataFrameModel can be defined without generics.
    """

    class Model(DataFrameModel):
        """Model Schema"""

        a: str = Field(str_startswith="B")
        b: int = Field(gt=6)
        c: float = Field()

    data_fail = [("Bread", 5, "Food"), ("Cutter", 15, 99.99)]

    spark_schema = T.StructType(
        [
            T.StructField("a", T.StringType(), False),  # should fail
            T.StructField("b", T.IntegerType(), False),  # should fail
            T.StructField("c", T.StringType(), False),  # should fail
        ],
    )

    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = Model.validate(check_obj=df_fail)
    assert df_out.pandera.errors is not None


def test_pyspark_bare_fields(spark):
    """
    Test schema and data level checks
    """

    class PanderaSchema(DataFrameModel):
        """Test schema"""

        id: T.IntegerType() = Field(gt=5)
        product_name: T.StringType() = Field(str_startswith="B")
        price: T.DecimalType(20, 5) = Field()
        description: T.ArrayType(T.StringType()) = Field()
        meta: T.MapType(T.StringType(), T.StringType()) = Field()

    data_fail = [
        (
            5,
            "Bread",
            44.4,
            ["description of product"],
            {"product_category": "dairy"},
        ),
        (
            15,
            "Butter",
            99.0,
            ["more details here"],
            {"product_category": "bakery"},
        ),
    ]

    spark_schema = T.StructType(
        [
            T.StructField("id", T.IntegerType(), False),
            T.StructField("product", T.StringType(), False),
            T.StructField("price", T.DecimalType(20, 5), False),
            T.StructField(
                "description", T.ArrayType(T.StringType(), False), False
            ),
            T.StructField(
                "meta", T.MapType(T.StringType(), T.StringType(), False), False
            ),
        ],
    )
    df_fail = spark_df(spark, data_fail, spark_schema)
    df_out = PanderaSchema.validate(check_obj=df_fail)
    assert df_out.pandera.errors is not None


def test_pyspark_fields_metadata():
    """
    Test schema and metadata on field
    """

    class PanderaSchema(DataFrameModel):
        """Pandera Schema Class"""

        id: T.IntegerType() = Field(
            gt=5,
            metadata={
                "usecase": ["telco", "retail"],
                "category": "product_pricing",
            },
        )
        product_name: T.StringType() = Field(str_startswith="B")
        price: T.DecimalType(20, 5) = Field()

        class Config:
            """Config of pandera class"""

            name = "product_info"
            strict = True
            coerce = True
            metadata = {"category": "product-details"}

    expected = {
        "product_info": {
            "columns": {
                "id": {
                    "usecase": ["telco", "retail"],
                    "category": "product_pricing",
                },
                "product_name": None,
                "price": None,
            },
            "dataframe": {"category": "product-details"},
        }
    }
    assert PanderaSchema.get_metadata() == expected


@pytest.mark.parametrize(
    "data, expectation",
    [
        (
            (),
            does_not_raise(),
        ),
        (
            ([1, 4], [2, 5], [3, 6]),
            does_not_raise(),
        ),
        (
            ([0, 0], [0, 0], [3, 6]),
            pytest.raises(pa.PysparkSchemaError),
        ),
    ],
    ids=["no_data", "unique_data", "duplicated_data"],
)
def test_dataframe_schema_unique(spark, data, expectation):
    """Test uniqueness checks on pyspark dataframes."""

    df = spark.createDataFrame(data, "a: int, b: int")

    # Test `unique` configuration with a single column
    class UniqueSingleColumn(pa.DataFrameModel):
        """Simple DataFrameModel containing a column."""

        a: T.IntegerType = pa.Field()
        b: T.IntegerType = pa.Field()

        class Config:
            """Config class."""

            unique = "a"

    assert isinstance(UniqueSingleColumn(df), DataFrame)

    with expectation:
        df_out = UniqueSingleColumn.validate(check_obj=df)
        if df_out.pandera.errors:
            print(f"{df_out.pandera.errors=}")
            raise pa.PysparkSchemaError

    # Test `unique` configuration with multiple columns
    class UniqueMultipleColumns(pa.DataFrameModel):
        """Simple DataFrameModel containing two columns."""

        a: T.IntegerType = pa.Field()
        b: T.IntegerType = pa.Field()

        class Config:
            """Config class."""

            unique = ["a", "b"]

    assert isinstance(UniqueMultipleColumns(df), DataFrame)

    with expectation:
        df_out = UniqueMultipleColumns.validate(check_obj=df)
        if df_out.pandera.errors:
            print(f"{df_out.pandera.errors=}")
            raise pa.PysparkSchemaError


@pytest.mark.parametrize(
    "unique_column_name",
    [
        "x",
        ["x", "y"],
        ["x", ""],
    ],
    ids=[
        "wrong_column",
        "multiple_wrong_columns",
        "multiple_wrong_columns_w_empty",
    ],
)
def test_dataframe_schema_unique_wrong_column(spark, unique_column_name):
    """Test uniqueness checks on pyspark dataframes."""

    df = spark.createDataFrame(([1, 2],), "a: int, b: int")

    # Test `unique` configuration with a single, wrongly named column
    class UniqueMultipleColumns(pa.DataFrameModel):
        """Simple DataFrameModel containing two columns."""

        a: T.IntegerType = pa.Field()
        b: T.IntegerType = pa.Field()

        class Config:
            """Config class."""

            unique = unique_column_name

    with pytest.raises(SchemaDefinitionError):
        _ = UniqueMultipleColumns.validate(check_obj=df)


def test_dataframe_schema_strict(spark, config_params: PanderaConfig) -> None:
    """
    Checks if strict=True whether a schema error is raised because either extra columns are present in the dataframe
    or missing columns in dataframe
    """
    if config_params.validation_depth != ValidationDepth.DATA_ONLY:
        schema = DataFrameSchema(
            {
                "a": pa.Column("long", nullable=True),
                "b": pa.Column("int", nullable=True),
            },
            strict=True,
        )
        df = spark.createDataFrame(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], ["a", "b", "c", "d"]
        )

        df_out = schema.validate(df.select(["a", "b"]))

        assert isinstance(df_out, DataFrame)

        with pytest.raises(pa.PysparkSchemaError):
            df_out = schema.validate(df)
            print(df_out.pandera.errors)
            if df_out.pandera.errors:
                raise pa.PysparkSchemaError

        schema.strict = "filter"
        assert isinstance(schema.validate(df), DataFrame)

        assert list(schema.validate(df).columns) == ["a", "b"]
        #
        with pytest.raises(pa.SchemaInitError):
            DataFrameSchema(
                {
                    "a": pa.Column(int, nullable=True),
                    "b": pa.Column(int, nullable=True),
                },
                strict="foobar",  # type: ignore[arg-type]
            )

        with pytest.raises(pa.PysparkSchemaError):
            df_out = schema.validate(df.select("a"))
            if df_out.pandera.errors:
                raise pa.PysparkSchemaError
        with pytest.raises(pa.PysparkSchemaError):
            df_out = schema.validate(df.select(["a", "c"]))
            if df_out.pandera.errors:
                raise pa.PysparkSchemaError


def test_docstring_substitution() -> None:
    """Test the docstring substitution decorator"""

    @docstring_substitution(
        test_substitution=test_docstring_substitution.__doc__
    )
    def function_expected():
        """%(test_substitution)s"""

    expected = test_docstring_substitution.__doc__
    assert function_expected.__doc__ == expected

    with pytest.raises(AssertionError) as exc_info:

        @docstring_substitution(
            test_docstring_substitution.__doc__,
            test_substitution=test_docstring_substitution.__doc__,
        )
        def function_expected():
            """%(test_substitution)s"""

    assert "Either positional args or keyword args are accepted" == str(
        exc_info.value
    )


# Define a fixture for the Schema
@pytest.fixture(scope="module", name="test_schema_optional_columns")
def test_schema():
    """Fixture containing DataFrameModel with optional columns."""

    class Schema(pa.DataFrameModel):
        """Simple DataFrameModel containing optional columns."""

        a: Optional[str]
        b: Optional[str] = pa.Field(eq="b")
        c: Optional[str]  # test pandera.typing alias

    return Schema


def test_optional_column(test_schema_optional_columns) -> None:
    """Test that optional columns are not required."""

    schema = test_schema_optional_columns.to_schema()
    assert not schema.columns[
        "a"
    ].required, "Optional column 'a' shouldn't be required"
    assert not schema.columns[
        "b"
    ].required, "Optional column 'b' shouldn't be required"
    assert not schema.columns[
        "c"
    ].required, "Optional column 'c' shouldn't be required"


def test_validation_succeeds_with_missing_optional_column(
    spark, test_schema_optional_columns
) -> None:
    """Test that validation succeeds even when an optional column is missing."""

    data = [("5", "b"), ("15", "b")]
    spark_schema = T.StructType(
        [
            T.StructField("a", T.StringType(), False),
            T.StructField("b", T.StringType(), False),
            # 'c' column is missing, but it's optional
        ],
    )
    df = spark_df(spark, data, spark_schema)
    df_out = test_schema_optional_columns.validate(check_obj=df)

    # `df_out.pandera.errors` should be empty if validation is successful.
    assert (
        df_out.pandera.errors == {}
    ), "No error should be raised in case of a missing optional column."


def test_invalid_field() -> None:
    """Test that invalid feilds raises a schemaInitError."""

    class Schema(DataFrameModel):  # pylint:disable=missing-class-docstring
        a: int = 0  # type: ignore[assignment]  # mypy identifies the wrong usage correctly

    with pytest.raises(
        pandera.errors.SchemaInitError,
        match="'a' can only be assigned a 'Field'",
    ):
        Schema.to_schema()


def test_registered_dataframemodel_checks(spark) -> None:
    """Check that custom registered checks work"""

    @pax.register_check_method(
        supported_types=DataFrame,
    )
    def always_true_check(df: DataFrame):
        # pylint: disable=unused-argument
        return True

    class ExampleDFModel(
        DataFrameModel
    ):  # pylint:disable=missing-class-docstring
        name: str
        age: int

        class Config:
            coerce = True
            always_true_check = ()

    example_data_cols = ("name", "age")
    example_data = [("foo", 42), ("bar", 24)]

    df = spark.createDataFrame(example_data, example_data_cols)

    out = ExampleDFModel.validate(df, lazy=False)

    assert not out.pandera.errors
