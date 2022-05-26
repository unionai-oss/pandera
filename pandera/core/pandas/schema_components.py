import warnings
from copy import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

import pandera.strategies as st
from pandera import errors
from pandera.core.base import BaseSchemaStrategyMixin
from pandera.core.pandas.array import ArraySchema
from pandera.core.pandas.schemas import DataFrameSchema
from pandera.core.pandas.types import (
    CheckList,
    PandasDtypeInputTypes,
    is_field,
    is_index,
    is_multiindex,
    is_table,
)
from pandera.error_formatters import scalar_failure_case
from pandera.errors import SchemaError, SchemaErrors


class ColumnStrategyMixin(BaseSchemaStrategyMixin):
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


class Column(ArraySchema, ColumnStrategyMixin):
    """Validate types and properties of DataFrame columns."""

    BACKEND = ...

    def __init__(
        self,
        dtype: PandasDtypeInputTypes = None,
        checks: CheckList = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: Union[str, Tuple[str, ...], None] = None,
        regex: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Create column validator object.

        :param dtype: datatype of the column. The datatype for type-checking
            a dataframe. If a string is specified, then assumes
            one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param unique: whether column values should be unique
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param required: Whether or not column is allowed to be missing
        :param name: column name in dataframe to validate.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a dataframe.
        :param title: A human-readable label for the column.
        :param description: An arbitrary textual description of the column.

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
            dtype,
            checks,
            nullable,
            unique,
            coerce,
            name,
            title,
            description,
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
            "checks": self._checks,
            "nullable": self._nullable,
            "unique": self._unique,
            "coerce": self._coerce,
            "required": self.required,
            "name": self._name,
            "regex": self._regex,
            "title": self.title,
            "description": self.description,
        }

    def set_name(self, name: str):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        self.name = name
        return self

    def coerce_dtype(self, obj: Union[pd.DataFrame, pd.Series, pd.Index]):
        """Coerce dtype of a column, handling duplicate column names."""
        # pylint: disable=super-with-arguments
        if is_field(obj) or is_index(obj):
            return super(Column, self).coerce_dtype(obj)
        return obj.apply(
            lambda x: super(Column, self).coerce_dtype(x), axis="columns"
        )

    def validate(
        self,
        check_obj: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Validate a Column in a DataFrame object.

        :param check_obj: pandas DataFrame to validate.
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
        if not inplace:
            check_obj = check_obj.copy()

        if self.name is None:
            raise errors.SchemaError(
                self,
                check_obj,
                "column name is set to None. Pass the ``name` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.",
            )

        def validate_column(check_obj, column_name):
            super(Column, copy(self).set_name(column_name)).validate(
                check_obj,
                head,
                tail,
                sample,
                random_state,
                lazy,
                inplace=inplace,
            )

        column_keys_to_check = (
            self.get_regex_columns(check_obj.columns)
            if self._regex
            else [self.name]
        )

        for column_name in column_keys_to_check:
            if self.coerce:
                check_obj[column_name] = self.coerce_dtype(
                    check_obj[column_name]
                )
            if is_table(check_obj[column_name]):
                for i in range(check_obj[column_name].shape[1]):
                    validate_column(
                        check_obj[column_name].iloc[:, [i]], column_name
                    )
            else:
                validate_column(check_obj, column_name)

        return check_obj

    def get_regex_columns(
        self, columns: Union[pd.Index, pd.MultiIndex]
    ) -> Union[pd.Index, pd.MultiIndex]:
        """Get matching column names based on regex column name pattern.

        :param columns: columns to regex pattern match
        :returns: matchin columns
        """
        if isinstance(self.name, tuple):
            # handle MultiIndex case
            if len(self.name) != columns.nlevels:
                raise IndexError(
                    f"Column regex name='{self.name}' is a tuple, expected a "
                    f"MultiIndex columns with {len(self.name)} number of "
                    f"levels, found {columns.nlevels} level(s)"
                )
            matches = np.ones(len(columns)).astype(bool)
            for i, name in enumerate(self.name):
                matched = pd.Index(
                    columns.get_level_values(i).astype(str).str.match(name)
                ).fillna(False)
                matches = matches & np.array(matched.tolist())
            column_keys_to_check = columns[matches]
        else:
            if is_multiindex(columns):
                raise IndexError(
                    f"Column regex name {self.name} is a string, expected a "
                    "dataframe where the index is a pd.Index object, not a "
                    "pd.MultiIndex object"
                )
            column_keys_to_check = columns[
                # str.match will return nan values when the index value is
                # not a string.
                pd.Index(columns.astype(str).str.match(self.name))
                .fillna(False)
                .tolist()
            ]
        if column_keys_to_check.shape[0] == 0:
            raise errors.SchemaError(
                self,
                columns,
                f"Column regex name='{self.name}' did not match any columns "
                "in the dataframe. Update the regex pattern so that it "
                f"matches at least one column:\n{columns.tolist()}",
            )
        # drop duplicates to account for potential duplicated columns in the
        # dataframe.
        return column_keys_to_check.drop_duplicates()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        def _compare_dict(obj):
            return {
                k: v if k != "_checks" else set(v)
                for k, v in obj.__dict__.items()
            }

        return _compare_dict(self) == _compare_dict(other)


class IndexStrategyMixin(BaseSchemaStrategyMixin):
    @st.strategy_import_error
    def strategy(self, *, size: int = None):
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

    def example(self, size: int = None) -> pd.Index:
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


class Index(ArraySchema, IndexStrategyMixin):
    """Validate types and properties of a DataFrame Index."""

    @property
    def names(self):
        """Get index names in the Index schema component."""
        return [self.name]

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def validate(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Validate DataFrameSchema or SeriesSchema Index.

        :check_obj: pandas DataFrame of Series containing index to validate.
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
        if is_multiindex(check_obj.index):
            raise errors.SchemaError(
                self, check_obj, "Attempting to validate mismatch index"
            )

        series_cls = pd.Series
        # NOTE: this is a hack to get pyspark.pandas working, this needs a more
        # principled implementation
        if type(check_obj).__module__ == "pyspark.pandas.frame":
            # pylint: disable=import-outside-toplevel
            import pyspark.pandas as ps

            series_cls = ps.Series

        if self.coerce:
            check_obj.index = self.coerce_dtype(check_obj.index)
            # handles case where pandas native string type is not supported
            # by index.
            obj_to_validate = self.dtype.coerce(
                series_cls(
                    check_obj.index.to_numpy(), name=check_obj.index.name
                )
            )
        else:
            obj_to_validate = series_cls(
                check_obj.index.to_numpy(), name=check_obj.index.name
            )

        assert is_field(
            super().validate(
                obj_to_validate,
                head,
                tail,
                sample,
                random_state,
                lazy,
                inplace,
            ),
        )
        return check_obj

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MultiIndex(DataFrameSchema):
    ...


def is_valid_multiindex_key(x: Tuple[Any, ...]) -> bool:
    """Check that a multi-index tuple key has all string elements"""
    return isinstance(x, tuple) and all(isinstance(i, str) for i in x)
