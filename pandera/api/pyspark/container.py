"""Core PySpark dataframe container specification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.backends.pyspark.register import register_pyspark_backends
from pandera.config import get_config_context
from pandera.dtypes import DataType
from pandera.engines import pyspark_engine
from pandera.utils import docstring_substitution

from .types import (
    PySparkDataFrameTypes,
    PySparkDtypeInputTypes,
    PySparkFrame,
)

if TYPE_CHECKING:
    import pandera.api.pyspark.components

N_INDENT_SPACES = 4


class DataFrameSchema(_DataFrameSchema[PySparkDataFrameTypes]):
    """A light-weight PySpark DataFrame validator."""

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        register_pyspark_backends()

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: PySparkDtypeInputTypes) -> None:
        """Set the pyspark dtype property."""
        self._dtype = pyspark_engine.Engine.dtype(value) if value else None

    def validate(
        self,
        check_obj: PySparkFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = True,
        inplace: bool = False,
    ) -> PySparkFrame:
        """Check if all columns in a dataframe have a column in the Schema.

        :param check_obj: DataFrame object i.e. the dataframe to be validated.
        :param head: Not used since spark has no concept of head or tail
        :param tail: Not used since spark has no concept of head or tail
        :param sample: validate a random sample of n% rows. Value ranges from
                0-1, for example 10% rows can be sampled using setting value as 0.1.
                refer below documentation.
                https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.sample.html
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated ``DataFrame``

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        Calling ``schema.validate`` returns the dataframe.


        .. doctest::
            :skipif: SKIP_PYSPARK_TYPING

            >>> import pandera.pyspark as psa
            >>> from pyspark.sql import SparkSession
            >>> import pyspark.sql.types as T
            >>> spark = SparkSession.builder.getOrCreate()
            >>> data = [("Bread", 9), ("Butter", 15)]
            >>> spark_schema = T.StructType(
            ...     [
            ...         T.StructField("product", T.StringType(), False),
            ...         T.StructField("price", T.IntegerType(), False),
            ...     ],
            ... )
            >>> df = spark.createDataFrame(data=data, schema=spark_schema)
            >>> schema_with_checks = psa.DataFrameSchema(
            ...     columns={
            ...         "product": psa.Column("str", checks=psa.Check.str_startswith("B")),
            ...         "price": psa.Column("int", checks=psa.Check.gt(5)),
            ...     },
            ...     name="product_schema",
            ...     description="schema for product info",
            ...     title="ProductSchema",
            ... )
            >>> schema_with_checks.validate(df).take(2)
        """
        if not get_config_context().validation_enabled:
            return check_obj

        return self.get_backend(check_obj).validate(
            check_obj=check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @docstring_substitution(validate_doc=_DataFrameSchema.__call__.__doc__)
    def __call__(
        self,
        dataframe: PySparkFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = True,
        inplace: bool = False,
    ) -> PySparkFrame:
        """%(validate_doc)s"""
        return self.validate(
            dataframe, head, tail, sample, random_state, lazy, inplace
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        def _compare_dict(obj):
            return {k: v for k, v in obj.__dict__.items() if k}

        return _compare_dict(self) == _compare_dict(other)

    #####################
    # Schema IO Methods #
    #####################

    def to_structtype(self) -> StructType:
        """Recover fields of DataFrameSchema as a Pyspark StructType object.

        As the output of this method will be used to specify a read schema in Pyspark
            (avoiding automatic schema inference), the False `nullable` properties are
            just ignored, as this check will be executed by the Pandera validations
            after a dataset is read.

        :returns: StructType object with current schema fields.
        """
        fields = [
            StructField(column, self.columns[column].dtype.type, True)
            for column in self.columns
        ]
        return StructType(fields)

    def to_ddl(self) -> str:
        """Recover fields of DataFrameSchema as a Pyspark DDL string.

        :returns: String with current schema fields, in compact DDL format.
        """
        # `StructType.toDDL()` is only available in internal java classes
        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("spark.sql.ansi.enabled", False)
        # Create a base dataframe from where we access underlying Java classes
        empty_df_with_schema = spark.createDataFrame([], self.to_structtype())

        return empty_df_with_schema._jdf.schema().toDDL()
