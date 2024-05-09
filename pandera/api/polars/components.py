"""Schema components for polars."""

from __future__ import annotations

import logging
from typing import Any, Optional

import polars as pl

from pandera.api.base.types import CheckList
from pandera.api.dataframe.components import ComponentSchema
from pandera.api.polars.types import PolarsCheckObjects, PolarsDtypeInputTypes
from pandera.backends.polars.register import register_polars_backends
from pandera.config import config_context, get_config_context
from pandera.engines import polars_engine
from pandera.utils import is_regex

logger = logging.getLogger(__name__)


class Column(ComponentSchema[PolarsCheckObjects]):
    """Polars column schema component."""

    def __init__(
        self,
        dtype: PolarsDtypeInputTypes = None,
        checks: Optional[CheckList] = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: Optional[str] = None,
        regex: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        metadata: Optional[dict] = None,
        drop_invalid_rows: bool = False,
        **column_kwargs,
    ) -> None:
        """Create column validator object.

        :param dtype: datatype of the column. The datatype for type-checking
            a dataframe. All `polars datatypes <https://docs.pola.rs/py-polars/html/reference/datatypes.html>`__,
            supported built-in python types that are supported by polars,
            and the pandera polars engine :ref:`datatypes <polars-dtypes>`.
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param unique: whether column values should be unique
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param required: Whether or not column is allowed to be missing
        :param name: column name in dataframe to validate. Names in the format
            '^{regex_pattern}$' are treated as regular expressions. During
            validation, this schema will be applied to any columns matching this
            pattern.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a dataframe. If the
            name is a regular expression, this attribute will automatically be
            set to True.
        :param title: A human-readable label for the column.
        :param description: An arbitrary textual description of the column.
        :param default: The default value for missing values in the column.
        :param metadata: An optional key value data.
        :param drop_invalid_rows: if True, drop invalid rows on validation.

        :raises SchemaInitError: if impossible to build schema from parameters

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "column": pa.Column(str)
        ... })
        >>>
        >>> schema.validate(pd.DataFrame({"column": ["foo", "bar"]}))
          column
        0    foo
        1    bar

        See :ref:`here<column>` for more usage details.
        """

        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            unique=unique,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
            **column_kwargs,
        )
        self.required = required
        self.regex = regex
        self.name = name

        self.set_regex()

    def _register_default_backends(self):
        register_polars_backends()

    def validate(
        self,
        check_obj: PolarsCheckObjects,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> PolarsCheckObjects:
        """Validate a Column in a DataFrame object.

        :param check_obj: polars LazyFrame to validate.
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
        :returns: validated DataFrame.
        """
        is_dataframe = isinstance(check_obj, pl.DataFrame)

        if is_dataframe:
            check_obj = check_obj.lazy()

        config_ctx = get_config_context(validation_depth_default=None)
        validation_depth = config_ctx.validation_depth
        with config_context(validation_depth=validation_depth):
            output = self.get_backend(check_obj).validate(
                check_obj,
                self,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
        return output

    @property
    def properties(self) -> dict[str, Any]:
        """Get column properties."""
        return {
            "dtype": self.dtype,
            "parsers": self.parsers,
            "checks": self.checks,
            "nullable": self.nullable,
            "unique": self.unique,
            "report_duplicates": self.report_duplicates,
            "coerce": self.coerce,
            "required": self.required,
            "name": self.name,
            "regex": self.regex,
            "title": self.title,
            "description": self.description,
            "default": self.default,
            "metadata": self.metadata,
        }

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = polars_engine.Engine.dtype(value) if value else None

    @property
    def selector(self):
        if self.name is not None and not is_regex(self.name) and self.regex:
            return f"^{self.name}$"
        return self.name

    def set_regex(self):
        if self.name is None:
            return

        if is_regex(self.name) and not self.regex:
            logger.info(
                f"Column schema '{self.name}' is a regex expression. "
                "Setting regex=True."
            )
            self.regex = True

    def set_name(self, name: str):
        """Set the name of the schema.

        If the name is a regex starting with '^' and ending with '$'
        set the regex attribute to True.
        """
        self.name = name
        self.set_regex()
        return self

    def strategy(self, *, size=None):
        """Create a ``hypothesis`` strategy for generating a Column.

        :param size: number of elements to generate
        :returns: a dataframe strategy for a single column.

        .. warning::

           This method is not implemented in the polars backend.
        """
        raise NotImplementedError(
            "Data synthesis is not supported in with polars schemas."
        )

    def strategy_component(self):
        """Generate column data object for use by DataFrame strategy.

        .. warning::

           This method is not implemented in the polars backend.
        """
        raise NotImplementedError(
            "Data synthesis is not supported in with polars schemas."
        )

    def example(self, size=None):
        """Generate an example of a particular size.

        :param size: number of elements in the generated Index.
        :returns: pandas DataFrame object.

        .. warning::

           This method is not implemented in the polars backend.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        raise NotImplementedError(
            "Data synthesis is not supported in with polars schemas."
        )
