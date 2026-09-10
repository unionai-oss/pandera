"""Common class for dataframe component specification."""

import copy
from typing import Any, Generic, Optional, TypeVar, cast

from pandera.api.base.schema import BaseSchema
from pandera.api.base.types import CheckList, ParserList
from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser
from pandera.dtypes import UniqueSettings
from pandera.engines import PYDANTIC_V2

if PYDANTIC_V2:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema


TComponentSchemaBase = TypeVar("TComponentSchemaBase", bound="ComponentSchema")
TDataObject = TypeVar("TDataObject")


class ComponentSchema(Generic[TDataObject], BaseSchema):
    """Base class for data container component, e.g. columns."""

    def __init__(
        self,
        dtype: Any | None = None,
        checks: CheckList | None = None,
        parsers: ParserList | None = None,
        nullable: bool = False,
        unique: bool = False,
        report_duplicates: UniqueSettings = "all",
        coerce: bool = False,
        name: Any = None,
        title: str | None = None,
        description: str | None = None,
        default: Any | None = None,
        metadata: dict | None = None,
        drop_invalid_rows: bool = False,
    ) -> None:
        """Initialize array schema.

        :param dtype: datatype of the column.
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a the data
            object (Series, DataFrame).
        :param nullable: Whether or not column can contain null values.
        :param unique: Whether or not column can contain duplicate
            values.
        :param report_duplicates: how to report unique errors
            - `exclude_first`: report all duplicates except first occurrence
            - `exclude_last`: report all duplicates except last occurrence
            - `all`: (default) report all duplicates
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param name: column name in dataframe to validate.
        :param title: A human-readable label for the series.
        :param description: An arbitrary textual description of the series.
        :param metadata: An optional key-value data.
        :param default: The default value for missing values in the series.
        :param drop_invalid_rows: if True, drop invalid rows on validation.
        """

        super().__init__(
            dtype=dtype,
            checks=checks,
            parsers=parsers,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
        )

        if parsers is None:
            parsers = []
        if isinstance(parsers, Parser):
            parsers = [parsers]

        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        self.parsers = parsers
        self.checks = checks
        self.nullable = nullable
        self.unique = unique
        self.report_duplicates = report_duplicates
        self.title = title
        self.description = description
        self.default = default

        self._validate_attributes()

    def _validate_attributes(self): ...

    @property
    def _allow_groupby(self):
        """Whether the schema or schema component allows groupby operations."""
        raise NotImplementedError(  # pragma: no cover
            "The _allow_groupby property must be implemented by subclasses "
            "of SeriesSchemaBase"
        )

    def coerce_dtype(self, check_obj: TDataObject) -> TDataObject:
        """Coerce type of the data by type specified in dtype.

        :param check_obj: data to coerce
        :returns: data of the same type as the input
        """
        return self.get_backend(check_obj).coerce_dtype(check_obj, schema=self)

    def validate(
        self,
        check_obj,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        """Validate a series or specific column in dataframe.

        :check_obj: data object to validate.
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
        :returns: validated DataFrame or Series.

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
        )

    def __call__(
        self,
        check_obj: TDataObject,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> TDataObject:
        """Alias for ``validate`` method."""
        return self.validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    if PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls._pydantic_validate,  # type: ignore[misc]
            )

    else:

        @classmethod
        def __get_validators__(cls):
            yield cls._pydantic_validate

    @classmethod
    def _pydantic_validate(  # type: ignore
        cls: TComponentSchemaBase, schema: Any
    ) -> TComponentSchemaBase:
        """Verify that the input is a compatible Schema."""
        if not isinstance(schema, cls):  # type: ignore
            raise TypeError(f"{schema} is not a {cls}.")

        return cast(TComponentSchemaBase, schema)

    #############################
    # Schema Transforms Methods #
    #############################

    def update_checks(self, checks: CheckList):
        """Create a new SeriesSchema with a new set of Checks

        :param checks: checks to set on the new schema
        :returns: a new SeriesSchema with a new set of checks
        """
        schema_copy = cast(ComponentSchema, copy.copy(self))
        schema_copy.checks = checks
        return schema_copy

    def set_checks(self, checks: CheckList):
        """Create a new SeriesSchema with a new set of Checks

        .. caution::
           This method will be deprecated in favor of ``update_checks`` in
           v0.15.0

        :param checks: checks to set on the new schema
        :returns: a new SeriesSchema with a new set of checks
        """
        return self.update_checks(checks)

    def __repr__(self):
        return (
            f"<Schema {self.__class__.__name__}"
            f"(name={self.name}, type={self.dtype!r})>"
        )
