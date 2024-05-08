"""Core pandas schema component specifications."""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import pandas as pd

import pandera.strategies as st
from pandera import errors
from pandera.api.base.types import CheckList, ParserList
from pandera.api.pandas.array import ArraySchema
from pandera.api.pandas.container import DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes
from pandera.dtypes import UniqueSettings


class Column(ArraySchema[pd.DataFrame]):
    """Validate types and properties of pandas DataFrame columns."""

    def __init__(
        self,
        dtype: PandasDtypeInputTypes = None,
        checks: Optional[CheckList] = None,
        parsers: Optional[ParserList] = None,
        nullable: bool = False,
        unique: bool = False,
        report_duplicates: UniqueSettings = "all",
        coerce: bool = False,
        required: bool = True,
        name: Union[str, Tuple[str, ...], None] = None,
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
        :param parsers: parsers to verify validity of the column
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
            parsers=parsers,
            checks=checks,
            nullable=nullable,
            unique=unique,
            report_duplicates=report_duplicates,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
        )
        if (
            name is not None
            and not isinstance(name, str)
            and not is_valid_multiindex_key(name)
            and regex
        ):
            raise ValueError(
                "You cannot specify a non-string name when setting regex=True"
            )
        self.required = required
        self.name = name
        self.regex = regex

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return True

    @property
    def properties(self) -> Dict[str, Any]:
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

    def set_name(self, name: str):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        self.name = name
        return self

    def get_regex_columns(self, check_obj) -> Iterable:
        """Get matching column names based on regex column name pattern.

        :param columns: columns to regex pattern match
        :returns: matchin columns
        """
        # pylint: disable=import-outside-toplevel
        from pandera.backends.pandas.components import ColumnBackend

        return cast(
            ColumnBackend, self.get_backend(check_type=pd.DataFrame)
        ).get_regex_columns(self, check_obj)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        def _compare_dict(obj):
            return {
                k: v if k not in ["_checks", "_parsers"] else set(v)
                for k, v in obj.__dict__.items()
            }

        return _compare_dict(self) == _compare_dict(other)

    ############################
    # Schema Transform Methods #
    ############################

    @st.strategy_import_error
    def strategy(self, *, size=None):
        """Create a ``hypothesis`` strategy for generating a Column.

        :param size: number of elements to generate
        :returns: a dataframe strategy for a single column.
        """
        return super().strategy(size=size).map(lambda x: x.to_frame())

    @st.strategy_import_error
    def strategy_component(self):
        """Generate column data object for use by DataFrame strategy."""
        return st.column_strategy(
            self.dtype,
            checks=self.checks,
            unique=self.unique,
            name=self.name,
        )

    def example(self, size=None) -> pd.DataFrame:
        """Generate an example of a particular size.

        :param size: number of elements in the generated Index.
        :returns: pandas DataFrame object.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return (
                super()
                .strategy(size=size)
                .example()
                .rename(self.name)
                .to_frame()
            )


class Index(ArraySchema[pd.Index]):
    """Validate types and properties of a pandas DataFrame Index."""

    @property
    def names(self):
        """Get index names in the Index schema component."""
        return [self.name]

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    ###########################
    # Schema Strategy Methods #
    ###########################

    @st.strategy_import_error
    def strategy(self, *, size: Optional[int] = None):
        """Create a ``hypothesis`` strategy for generating an Index.

        :param size: number of elements to generate.
        :returns: index strategy.
        """
        return st.index_strategy(
            self.dtype,  # type: ignore
            checks=self.checks,
            nullable=self.nullable,
            unique=self.unique,
            name=self.name,
            size=size,
        )

    @st.strategy_import_error
    def strategy_component(self):
        """Generate column data object for use by MultiIndex strategy."""
        return st.column_strategy(
            self.dtype,
            checks=self.checks,
            unique=self.unique,
            name=self.name,
        )

    def example(self, size: Optional[int] = None) -> pd.Index:
        """Generate an example of a particular size.

        :param size: number of elements in the generated Index.
        :returns: pandas Index object.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(size=size).example()


class MultiIndex(DataFrameSchema):
    """Validate types and properties of a pandas DataFrame MultiIndex.

    This class inherits from :class:`~pandera.api.pandas.container.DataFrameSchema` to
    leverage its validation logic.
    """

    def __init__(
        self,
        indexes: List[Index],
        coerce: bool = False,
        strict: bool = False,
        name: Optional[str] = None,
        ordered: bool = True,
        unique: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Create MultiIndex validator.

        :param indexes: list of Index validators for each level of the
            MultiIndex index.
        :param coerce: Whether or not to coerce the MultiIndex to the
            specified dtypes before validation
        :param strict: whether or not to accept columns in the MultiIndex that
            aren't defined in the ``indexes`` argument.
        :param name: name of schema component
        :param ordered: whether or not to validate the indexes order.
        :param unique: a list of index names that should be jointly unique.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>>
        >>> schema = pa.DataFrameSchema(
        ...     columns={"column": pa.Column(int)},
        ...     index=pa.MultiIndex([
        ...         pa.Index(str,
        ...               pa.Check(lambda s: s.isin(["foo", "bar"])),
        ...               name="index0"),
        ...         pa.Index(int, name="index1"),
        ...     ])
        ... )
        >>>
        >>> df = pd.DataFrame(
        ...     data={"column": [1, 2, 3]},
        ...     index=pd.MultiIndex.from_arrays(
        ...         [["foo", "bar", "foo"], [0, 1, 2]],
        ...         names=["index0", "index1"],
        ...     )
        ... )
        >>>
        >>> schema.validate(df)
                       column
        index0 index1
        foo    0            1
        bar    1            2
        foo    2            3

        See :ref:`here<multiindex>` for more usage details.

        """
        if any(not isinstance(i, Index) for i in indexes):
            raise errors.SchemaInitError(
                f"expected a list of Index objects, found {indexes} "
                f"of type {[type(x) for x in indexes]}"
            )
        self.indexes = indexes
        columns = {}
        for i, index in enumerate(indexes):
            if not ordered and index.name is None:
                # if the MultiIndex is not ordered, there's no way of
                # determining how to get the index level without an explicit
                # index name
                raise errors.SchemaInitError(
                    "You must specify index names if MultiIndex schema "
                    "component is not ordered."
                )
            columns[i if index.name is None else index.name] = Column(
                dtype=index._dtype,
                checks=index.checks,
                nullable=index.nullable,
                unique=index.unique,
            )
        super().__init__(
            columns=columns,
            coerce=coerce,
            strict=strict,
            name=name,
            ordered=ordered,
            unique=unique,
        )

    @property
    def names(self):
        """Get index names in the MultiIndex schema component."""
        return [index.name for index in self.indexes]

    @property
    def coerce(self):
        """Whether or not to coerce data types."""
        return self._coerce or any(index.coerce for index in self.indexes)

    @coerce.setter
    def coerce(self, value: bool) -> None:
        """Set coerce attribute."""
        self._coerce = value

    def __repr__(self):
        return (
            f"<Schema {self.__class__.__name__}("
            f"indexes={self.indexes}, "
            f"coerce={self.coerce}, "
            f"strict={self.strict}, "
            f"name={self.name}, "
            f"ordered={self.ordered}"
            ")>"
        )

    def __str__(self):
        indent = " " * 4

        indexes_str = "[\n"
        for index in self.indexes:
            indexes_str += f"{indent * 2}{index}\n"
        indexes_str += f"{indent}]"

        return (
            f"<Schema {self.__class__.__name__}(\n"
            f"{indent}indexes={indexes_str}\n"
            f"{indent}coerce={self.coerce},\n"
            f"{indent}strict={self.strict},\n"
            f"{indent}name={self.name},\n"
            f"{indent}ordered={self.ordered}\n"
            ")>"
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    ###########################
    # Schema Strategy Methods #
    ###########################

    @st.strategy_import_error
    # NOTE: remove these ignore statements as part of
    # https://github.com/pandera-dev/pandera/issues/403
    # pylint: disable=arguments-differ
    def strategy(self, *, size=None):  # type: ignore
        return st.multiindex_strategy(indexes=self.indexes, size=size)

    # NOTE: remove these ignore statements as part of
    # https://github.com/pandera-dev/pandera/issues/403
    # pylint: disable=arguments-differ
    def example(self, size=None) -> pd.MultiIndex:  # type: ignore
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(size=size).example()


def is_valid_multiindex_key(x: Tuple[Any, ...]) -> bool:
    """Check that a multi-index tuple key has all string elements"""
    return isinstance(x, tuple) and all(isinstance(i, str) for i in x)
