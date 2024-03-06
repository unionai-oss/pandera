"""Schema components for polars."""

from typing import Any, Optional, cast

from pandera.api.base.types import CheckList
from pandera.api.pandas.components import Column as _Column
from pandera.api.polars.types import PolarsDtypeInputTypes
from pandera.engines import polars_engine
from pandera.utils import is_regex


class Column(_Column):
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
    ) -> None:
        """Create column validator object.

        :param dtype: datatype of the column. The datatype for type-checking
            a dataframe. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param unique: whether column values should be unique
        :param report_duplicates: how to report unique errors
            - `exclude_first`: report all duplicates except first occurence
            - `exclude_last`: report all duplicates except last occurence
            - `all`: (default) report all duplicates
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param required: Whether or not column is allowed to be missing
        :param name: column name in dataframe to validate.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a dataframe.
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
            required=required,
            name=name,
            regex=regex,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
        )
        if self.name is not None:
            self.set_name(cast(str, name))

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = polars_engine.Engine.dtype(value) if value else None

    def set_name(self, name: str):
        """Used to set or modify the name of a column object.

        This method automatically handles regex columns as defined by polars
        regex column selection:
        https://docs.pola.rs/user-guide/expressions/column-selections/#by-regular-expressions

        - If regex is False but the supplied name is a regex pattern, then the
          regex attribute is automatically set to True.
        - If regex is True but the supplied name is not a regex pattern, then
          the name is automatically converted to a regex pattern.
        - Otherwise just set the name attribute to the supplied name.

        :param str name: the name of the column object
        """
        if is_regex(name) and not self.regex:
            self.regex = True
        elif not is_regex(name) and self.regex:
            name = f"^{name}$"

        self.name = name
        return self
