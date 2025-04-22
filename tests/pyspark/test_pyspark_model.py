"""Unit tests for DataFrameModel module."""

# pylint:disable=abstract-method

from contextlib import nullcontext as does_not_raise
from typing import Optional

import pyspark.sql.types as T
import pytest
from pyspark.sql import DataFrame

import pandera
import pandera.api.extensions as pax
import pandera.pyspark as pa
from pandera.api.pyspark.model import docstring_substitution
from pandera.config import PanderaConfig, ValidationDepth
from pandera.errors import SchemaDefinitionError
from pandera.pyspark import DataFrameModel, DataFrameSchema, Field
from tests.pyspark.conftest import spark_df

pytestmark = pytest.mark.parametrize(
    "spark_session", ["spark", "spark_connect"]
)


def test_schema_with_bare_types(
    spark_session,  # pylint:disable=unused-argument
):
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


def test_schema_with_bare_types_and_field(
    spark_session,  # pylint:disable=unused-argument
):
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


def test_schema_with_bare_types_field_and_checks(spark_session, request):
    """
    Test that DataFrameModel can be defined without generics.
    """
    spark = request.getfixturevalue(spark_session)

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


def test_schema_with_bare_types_field_type(spark_session, request):
    """
    Test that DataFrameModel can be defined without generics.
    """
    spark = request.getfixturevalue(spark_session)

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


def test_pyspark_bare_fields(spark_session, request):
    """
    Test schema and data level checks
    """
    spark = request.getfixturevalue(spark_session)

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


def test_pyspark_fields_metadata(
    spark_session,  # pylint:disable=unused-argument
):
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
def test_dataframe_schema_unique(spark_session, data, expectation, request):
    """Test uniqueness checks on pyspark dataframes."""
    spark = request.getfixturevalue(spark_session)
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
def test_dataframe_schema_unique_wrong_column(
    spark_session, unique_column_name, request
):
    """Test uniqueness checks on pyspark dataframes."""
    spark = request.getfixturevalue(spark_session)
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


def test_dataframe_schema_strict(
    spark_session, config_params: PanderaConfig, request
) -> None:
    """
    Checks if strict=True whether a schema error is raised because either extra columns are present in the dataframe
    or missing columns in dataframe
    """
    spark = request.getfixturevalue(spark_session)
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


def test_docstring_substitution(
    spark_session,  # pylint:disable=unused-argument
) -> None:
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


def test_optional_column(
    test_schema_optional_columns,
    spark_session,  # pylint:disable=unused-argument
) -> None:
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
    spark_session, test_schema_optional_columns, request
) -> None:
    """Test that validation succeeds even when an optional column is missing."""
    spark = request.getfixturevalue(spark_session)
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


def test_invalid_field(
    spark_session,  # pylint:disable=unused-argument
) -> None:
    """Test that invalid fields raises a schemaInitError."""

    class Schema(DataFrameModel):  # pylint:disable=missing-class-docstring
        a: int = 0  # type: ignore[assignment]  # mypy identifies the wrong usage correctly

    with pytest.raises(
        pandera.errors.SchemaInitError,
        match="'a' can only be assigned a 'Field'",
    ):
        Schema.to_schema()


# For the second parameterized `spark_session` run, `@pax.register_check_method` will
# raise a ValueError due to a duplicated registration tentative
@pytest.mark.xfail(raises=ValueError)
def test_registered_dataframemodel_checks(spark_session, request) -> None:
    """Check that custom registered checks work"""
    spark = request.getfixturevalue(spark_session)

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


@pytest.fixture(scope="function")
def model_with_datatypes(
    spark_session,  # pylint:disable=unused-argument
):
    """
    Model containing all common datatypes for PySpark namespace.
    """

    class SchemaWithDatatypes(DataFrameModel):
        non_nullable: T.IntegerType = Field(nullable=False)
        binary: T.BinaryType = Field()
        byte: T.ByteType = Field()
        text: T.StringType = Field()
        integer: T.IntegerType = Field()
        long: T.LongType = Field()
        float: T.FloatType = Field()
        double: T.DoubleType = Field()
        boolean: T.BooleanType = Field()
        decimal: T.DecimalType = Field()
        date: T.DateType = Field()
        timestamp: T.TimestampType = Field()
        timestamp_ntz: T.TimestampNTZType = Field()
        array: T.ArrayType(T.StringType()) = Field()
        map: T.MapType(T.StringType(), T.IntegerType()) = Field()
        nested_structure: T.MapType(
            T.ArrayType(T.StringType()),
            T.MapType(T.StringType(), T.ArrayType(T.StringType())),
        ) = Field()

    return SchemaWithDatatypes


@pytest.fixture(scope="function")
def model_with_multiple_parent_classes(
    spark_session,  # pylint:disable=unused-argument
):
    """
    Model inherited from multiple parent classes.
    """

    class BaseClassA1(DataFrameModel):
        byte: T.ByteType = Field()
        text: T.StringType = Field()
        array: T.ArrayType(T.StringType()) = Field()

    class BaseClassA2(DataFrameModel):
        non_nullable: T.IntegerType = Field(nullable=False)
        text: T.StringType = Field()
        integer: T.IntegerType = Field()
        map: T.MapType(T.StringType(), T.IntegerType()) = Field()

    class BaseClassB(BaseClassA1, BaseClassA2):
        array: T.ArrayType(T.IntegerType()) = Field()
        map: T.MapType(T.IntegerType(), T.DoubleType()) = Field()

    class BaseClassC(DataFrameModel):
        text_new: T.StringType = Field()

    class BaseClassFinal(BaseClassB, BaseClassC):
        # Notes:
        # - B overwrites the types annotations for `array` and `map`
        # - `text` is duplicated between A1 and A2
        # - Adding a new field in C
        pass

    return BaseClassFinal


def test_schema_to_structtype(
    model_with_datatypes,
    spark_session,  # pylint:disable=unused-argument
):
    """
    Test the conversion from a model to a StructType object through `to_structtype()`.
    """

    assert model_with_datatypes.to_structtype() == T.StructType(
        [
            T.StructField(
                name="non_nullable", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(
                name="binary", dataType=T.BinaryType(), nullable=True
            ),
            T.StructField(name="byte", dataType=T.ByteType(), nullable=True),
            T.StructField(name="text", dataType=T.StringType(), nullable=True),
            T.StructField(
                name="integer", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(name="long", dataType=T.LongType(), nullable=True),
            T.StructField(name="float", dataType=T.FloatType(), nullable=True),
            T.StructField(
                name="double", dataType=T.DoubleType(), nullable=True
            ),
            T.StructField(
                name="boolean", dataType=T.BooleanType(), nullable=True
            ),
            T.StructField(
                name="decimal", dataType=T.DecimalType(), nullable=True
            ),
            T.StructField(name="date", dataType=T.DateType(), nullable=True),
            T.StructField(
                name="timestamp", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="timestamp_ntz", dataType=T.TimestampType(), nullable=True
            ),
            T.StructField(
                name="array",
                dataType=T.ArrayType(T.StringType()),
                nullable=True,
            ),
            T.StructField(
                name="map",
                dataType=T.MapType(T.StringType(), T.IntegerType()),
                nullable=True,
            ),
            T.StructField(
                name="nested_structure",
                dataType=T.MapType(
                    T.ArrayType(T.StringType()),
                    T.MapType(T.StringType(), T.ArrayType(T.StringType())),
                ),
                nullable=True,
            ),
        ]
    )


def test_schema_to_ddl(
    model_with_datatypes,
    spark_session,  # pylint:disable=unused-argument
):
    """
    Test the conversion from a model to a DDL string through `to_ddl()`.
    """

    assert model_with_datatypes.to_ddl() == ",".join(
        [
            "non_nullable INT",
            "binary BINARY",
            "byte TINYINT",
            "text STRING",
            "integer INT",
            "long BIGINT",
            "float FLOAT",
            "double DOUBLE",
            "boolean BOOLEAN",
            "decimal DECIMAL(10,0)",
            "date DATE",
            "timestamp TIMESTAMP",
            "timestamp_ntz TIMESTAMP",
            "array ARRAY<STRING>",
            "map MAP<STRING, INT>",
            "nested_structure MAP<ARRAY<STRING>, MAP<STRING, ARRAY<STRING>>>",
        ]
    )


def test_inherited_schema_to_structtype(
    model_with_multiple_parent_classes,
    spark_session,  # pylint:disable=unused-argument
):
    """
    Test the final inheritance for a model with a longer parent class structure.
    """

    assert model_with_multiple_parent_classes.to_structtype() == T.StructType(
        [
            T.StructField(
                name="text_new", dataType=T.StringType(), nullable=True
            ),  # A new field was kept
            T.StructField(
                name="non_nullable", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(
                name="text", dataType=T.StringType(), nullable=True
            ),  # Only one `text` was kept
            T.StructField(
                name="integer", dataType=T.IntegerType(), nullable=True
            ),
            T.StructField(
                name="map",
                dataType=T.MapType(T.IntegerType(), T.DoubleType()),
                nullable=True,
            ),  # `map` has the overloaded `IntegerType/DoubleType`
            T.StructField(name="byte", dataType=T.ByteType(), nullable=True),
            T.StructField(
                name="array",
                dataType=T.ArrayType(T.IntegerType()),
                nullable=True,
            ),  # `array` has the overloaded `IntegerType`
        ]
    )
