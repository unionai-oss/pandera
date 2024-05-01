"""Core pyspark column specification."""

import copy
from typing import Any, List, Optional, TypeVar, cast

import pyspark.sql as ps

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.base.schema import BaseSchema, inferred_schema_guard
from pandera.api.checks import Check
from pandera.api.pyspark.types import CheckList, PySparkDtypeInputTypes
from pandera.backends.pyspark.register import register_pyspark_backends
from pandera.dtypes import DataType
from pandera.engines import pyspark_engine

TColumnSchemaBase = TypeVar("TColumnSchemaBase", bound="ColumnSchema")


class ColumnSchema(BaseSchema):
    """Base column validator object."""

    def __init__(
        self,
        dtype: Optional[PySparkDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        nullable: bool = False,
        coerce: bool = False,
        name: Any = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Initialize column schema.

        :param dtype: datatype of the column.
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a
            dataframe object.
        :param nullable: Whether or not column can contain null values.
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param name: column name in dataframe to validate.
        :param title: A human-readable label for the column.
        :param description: An arbitrary textual description of the series.
        :param metadata: An optional key-value data element.
        :type nullable: bool
        """

        super().__init__(
            dtype=dtype,
            checks=checks,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )

        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self.checks = checks
        self.nullable = nullable
        self.title = title
        self.description = description
        self.metadata = metadata

    def _register_default_backends(self):
        register_pyspark_backends()

    @property
    def dtype(self) -> DataType:
        """Get the pyspark dtype"""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: Optional[PySparkDtypeInputTypes]) -> None:
        """Set the pyspark dtype"""
        self._dtype = (
            pyspark_engine.Engine.dtype(value) if value else None
        )  # pylint:disable=no-value-for-parameter

    def validate(
        self,
        check_obj,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        error_handler: ErrorHandler = None,
    ):
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Validate a specific column in a dataframe.

        :check_obj: pyspark DataFrame to validate.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :param error_handler: pyspark error handler object to provide the error in a dictionary format.
        :returns: validated DataFrame.

        """
        return self.get_backend(check_obj).validate(
            check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
            error_handler=error_handler,
        )

    def __call__(
        self,
        check_obj: ps.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        """Alias for ``validate`` method."""
        return self.validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validate

    @classmethod
    def _pydantic_validate(  # type: ignore
        cls: TColumnSchemaBase, schema: Any
    ) -> TColumnSchemaBase:
        """Verify that the input is a compatible Schema."""
        if not isinstance(schema, cls):  # type: ignore
            raise TypeError(f"{schema} is not a {cls}.")

        return cast(TColumnSchemaBase, schema)

    #############################
    # Schema Transforms Methods #
    #############################

    @inferred_schema_guard
    def update_checks(self, checks: List[Check]):
        """Create a new Schema with a new set of Checks

        :param checks: checks to set on the new schema
        :returns: a new Schema with a new set of checks
        """
        schema_copy = cast(ColumnSchema, copy.deepcopy(self))
        schema_copy.checks = checks
        return schema_copy

    def set_checks(self, checks: CheckList):
        """Create a new Schema with a new set of Checks

        .. caution::
           This method will be deprecated in favor of ``update_checks`` in
           v0.15.0

        :param checks: checks to set on the new schema
        :returns: a new Schema with a new set of checks
        """
        return self.update_checks(checks)

    def __repr__(self):
        return (
            f"<Schema {self.__class__.__name__}"
            f"(name={self.name}, type={self.dtype!r})>"
        )

    def __str__(self):
        return f"column '{self.name}' with type {self.dtype}"
