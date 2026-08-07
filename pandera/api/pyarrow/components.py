"""Core PyArrow schema component specifications."""

from __future__ import annotations

import logging
from typing import Any

from pandera.api.base.types import CheckList
from pandera.api.dataframe.components import ComponentSchema
from pandera.api.pyarrow.types import (
    PyArrowCheckObjects,
    PyArrowDtypeInputTypes,
)
from pandera.api.pyarrow.utils import resolve_dtype
from pandera.backends.pyarrow.register import register_pyarrow_backends
from pandera.utils import is_regex

logger = logging.getLogger(__name__)


class Column(ComponentSchema[PyArrowCheckObjects]):
    """Validate types and properties of pyarrow table columns."""

    def __init__(
        self,
        dtype: PyArrowDtypeInputTypes | None = None,
        checks: CheckList | None = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: str | None = None,
        regex: bool = False,
        title: str | None = None,
        description: str | None = None,
        default: Any | None = None,
        metadata: dict | None = None,
        drop_invalid_rows: bool = False,
        **column_kwargs,
    ) -> None:
        """Create column validator object.

        :param dtype: datatype of the column. Accepts native ``pyarrow``
            types (e.g. ``pyarrow.int64()``), narwhals dtypes, the pandera
            abstract datatypes, supported python builtins (``int``, ``float``,
            ``str``, ``bool``) and their string aliases.
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param unique: whether column values should be unique
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param required: Whether or not column is allowed to be missing
        :param name: column name in the table to validate. Names in the format
            '^{regex_pattern}$' are treated as regular expressions. During
            validation, this schema will be applied to any columns matching
            this pattern.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a table.
        :param title: A human-readable label for the column.
        :param description: An arbitrary textual description of the column.
        :param default: The default value for missing values in the column.
        :param metadata: An optional key value data.
        :param drop_invalid_rows: if True, drop invalid rows on validation.

        :raises SchemaInitError: if impossible to build schema from parameters

        :example:

        >>> import pyarrow
        >>> import pandera.pyarrow as pa
        >>>
        >>> schema = pa.DataFrameSchema({"column": pa.Column(str)})
        >>> schema.validate(pyarrow.table({"column": ["foo", "bar"]}))
        pyarrow.Table
        column: string
        ----
        column: [["foo","bar"]]
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

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        register_pyarrow_backends()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = resolve_dtype(value)

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
        """Set or modify the name of a column object.

        :param str name: the name of the column object
        """
        self.name = name
        self.set_regex()
        return self

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

    def strategy(self, *, size=None):
        """Data synthesis is not supported for pyarrow schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with pyarrow schemas."
        )

    def strategy_component(self):
        """Data synthesis is not supported for pyarrow schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with pyarrow schemas."
        )

    def example(self, size=None):
        """Data synthesis is not supported for pyarrow schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with pyarrow schemas."
        )
