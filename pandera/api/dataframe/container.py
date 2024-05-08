"""Common class for dataframe schema objects."""

from __future__ import annotations

import copy
import os
import sys
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from pandera import errors
from pandera import strategies as st
from pandera.api.base.schema import BaseSchema, inferred_schema_guard
from pandera.api.base.types import CheckList, ParserList, StrictType
from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser
from pandera.dtypes import DataType, UniqueSettings
from pandera.engines import PYDANTIC_V2

# if python version is < 3.11, import Self from typing_extensions
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


TDataObject = TypeVar("TDataObject")


if PYDANTIC_V2:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema

N_INDENT_SPACES = 4


# pylint: disable=too-many-public-methods
class DataFrameSchema(Generic[TDataObject], BaseSchema):
    def __init__(
        self,
        columns: Optional[Dict[Any, Any]] = None,
        checks: Optional[CheckList] = None,
        parsers: Optional[ParserList] = None,
        index=None,
        dtype: Optional[Any] = None,
        coerce: bool = False,
        strict: StrictType = False,
        name: Optional[str] = None,
        ordered: bool = False,
        unique: Optional[Union[str, List[str]]] = None,
        report_duplicates: UniqueSettings = "all",
        unique_column_names: bool = False,
        add_missing_columns: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        drop_invalid_rows: bool = False,
    ) -> None:
        """Initialize DataFrameSchema validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :type columns: mapping of column names and column schema component.
        :param checks: dataframe-wide checks.
        :param parsers: dataframe-wide parsers.
        :param index: specify the datatypes and properties of the index.
        :param dtype: datatype of the dataframe. This overrides the data
            types specified in any of the columns. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes.
        :param coerce: whether or not to coerce all of the columns on
            validation. This overrides any coerce setting at the column
            or index level. This has no effect on columns where
            ``dtype=None``.
        :param strict: ensure that all and only the columns defined in the
            schema are present in the dataframe. If set to 'filter',
            only the columns in the schema will be passed to the validated
            dataframe. If set to filter and columns defined in the schema
            are not present in the dataframe, will throw an error.
        :param name: name of the schema.
        :param ordered: whether or not to validate the columns order.
        :param unique: a list of columns that should be jointly unique.
        :param report_duplicates: how to report unique errors
            - `exclude_first`: report all duplicates except first occurence
            - `exclude_last`: report all duplicates except last occurence
            - `all`: (default) report all duplicates
        :param unique_column_names: whether or not column names must be unique.
        :param add_missing_columns: add missing column names with either default
            value, if specified in column schema, or NaN if column is nullable.
        :param title: A human-readable label for the schema.
        :param description: An arbitrary textual description of the schema.
        :param metadata: An optional key-value data.
        :param drop_invalid_rows: if True, drop invalid rows on validation.

        :raises SchemaInitError: if impossible to build schema from parameters

        :examples:

        >>> import pandera as pa
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "str_column": pa.Column(str),
        ...     "float_column": pa.Column(float),
        ...     "int_column": pa.Column(int),
        ...     "date_column": pa.Column(pa.DateTime),
        ... })

        Use the pandas API to define checks, which takes a function with
        the signature: ``pd.Series -> Union[bool, pd.Series]`` where the
        output series contains boolean values.

        >>> schema_withchecks = pa.DataFrameSchema({
        ...     "probability": pa.Column(
        ...         float, pa.Check(lambda s: (s >= 0) & (s <= 1))),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": pa.Column(
        ...         str, [
        ...             pa.Check(lambda s: s.isin(["dog", "cat", "duck"])),
        ...             pa.Check(lambda s: (s == "dog").mean() > 0.5),
        ...         ]),
        ... })

        See :ref:`here<DataFrameSchemas>` for more usage details.

        """

        if columns is None:
            columns = {}

        _validate_columns(columns)
        columns = _columns_renamed(columns)

        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        if parsers is None:
            parsers = []
        if isinstance(parsers, Parser):
            parsers = [parsers]

        self._dtype: Optional[DataType] = None

        super().__init__(
            dtype=dtype,
            checks=checks,
            parsers=parsers,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )

        self.columns: Dict[Any, Any] = (  # type: ignore [name-defined]
            {} if columns is None else columns
        )

        self.index = index
        self.strict: Union[bool, str] = strict
        self._coerce = coerce
        self.ordered = ordered
        self._unique = unique
        self.report_duplicates = report_duplicates
        self.unique_column_names = unique_column_names
        self.add_missing_columns = add_missing_columns
        self.drop_invalid_rows = drop_invalid_rows

        # this attribute is not meant to be accessed by users and is explicitly
        # set to True in the case that a schema is created by infer_schema.
        self._IS_INFERRED = False
        self.metadata = metadata

        self._validate_attributes()

    def _validate_attributes(self):
        if self.strict not in (False, True, "filter"):
            raise errors.SchemaInitError(
                "strict parameter must equal either `True`, `False`, "
                "or `'filter'`."
            )

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: Any) -> None:
        """Set the pandas dtype property."""
        raise NotImplementedError

    @property
    def coerce(self) -> bool:
        """Whether to coerce series to specified type."""
        if isinstance(self.dtype, DataType):
            return self.dtype.auto_coerce or self._coerce
        return self._coerce

    @coerce.setter
    def coerce(self, value: bool) -> None:
        """Set coerce attribute"""
        self._coerce = value

    @property
    def unique(self):
        """List of columns that should be jointly unique."""
        return self._unique

    @unique.setter
    def unique(self, value: Optional[Union[str, List[str]]]) -> None:
        """Set unique attribute."""
        self._unique = [value] if isinstance(value, str) else value

    # the _is_inferred getter and setter methods are not public
    @property
    def _is_inferred(self) -> bool:
        return self._IS_INFERRED

    @_is_inferred.setter
    def _is_inferred(self, value: bool) -> None:
        self._IS_INFERRED = value

    @property
    def dtypes(self) -> Dict[str, DataType]:
        # pylint:disable=anomalous-backslash-in-string
        """
        A dict where the keys are column names and values are
        :class:`~pandera.dtypes.DataType` s for the column. Excludes columns
        where `regex=True`.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_columns = [
            name for name, col in self.columns.items() if col.regex
        ]
        if regex_columns:
            warnings.warn(
                "Schema has columns specified as regex column names: "
                f"{regex_columns}. Use the `get_dtypes` to get the datatypes "
                "for these columns.",
                UserWarning,
            )
        return {n: c.dtype for n, c in self.columns.items() if not c.regex}

    def get_metadata(self) -> Optional[dict]:
        """Provide metadata for columns and schema level"""
        res: Dict[Any, Any] = {"columns": {}}
        for k in self.columns.keys():
            res["columns"][k] = self.columns[k].properties["metadata"]

        res["dataframe"] = self.metadata

        meta = {}
        meta[self.name] = res
        return meta

    def get_dtypes(self, check_obj: TDataObject) -> Dict[str, DataType]:
        """
        Same as the ``dtype`` property, but expands columns where
        ``regex == True`` based on the supplied dataframe.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_dtype = {}
        for _, column in self.columns.items():
            backend = column.get_backend(check_obj)
            if column.regex:
                regex_dtype.update(
                    {
                        c: column.dtype
                        for c in backend.get_regex_columns(column, check_obj)
                    }
                )
        return {
            **{n: c.dtype for n, c in self.columns.items() if not c.regex},
            **regex_dtype,
        }

    def coerce_dtype(self, check_obj: TDataObject) -> TDataObject:
        return self.get_backend(check_obj).coerce_dtype(check_obj, schema=self)

    def validate(
        self,
        check_obj: TDataObject,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> TDataObject:
        raise NotImplementedError

    def __call__(
        self,
        dataframe: TDataObject,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> TDataObject:
        """Alias for :func:`DataFrameSchema.validate` method.

        :param pd.DataFrame dataframe: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :type head: int
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :type tail: int
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        """
        return self.validate(
            dataframe, head, tail, sample, random_state, lazy, inplace
        )

    def __repr__(self) -> str:
        """Represent string for logging."""
        return (
            f"<Schema {self.__class__.__name__}("
            f"columns={self.columns}, "
            f"checks={self.checks}, "
            f"parsers={self.parsers}, "
            f"index={self.index.__repr__()}, "
            f"coerce={self.coerce}, "
            f"dtype={self._dtype}, "
            f"strict={self.strict}, "
            f"name={self.name}, "
            f"ordered={self.ordered}, "
            f"unique_column_names={self.unique_column_names}"
            f"metadata='{self.metadata}, "
            f"unique_column_names={self.unique_column_names}, "
            f"add_missing_columns={self.add_missing_columns}"
            ")>"
        )

    def __str__(self) -> str:
        """Represent string for user inspection."""

        def _format_multiline(json_str, arg):
            return "\n".join(
                f"{indent}{line}" if i != 0 else f"{indent}{arg}={line}"
                for i, line in enumerate(json_str.split("\n"))
            )

        indent = " " * N_INDENT_SPACES
        if self.columns:
            columns_str = f"{indent}columns={{\n"
            for colname, col in self.columns.items():
                columns_str += f"{indent * 2}'{colname}': {col}\n"
            columns_str += f"{indent}}}"
        else:
            columns_str = f"{indent}columns={{}}"

        if self.checks:
            checks_str = f"{indent}checks=[\n"
            for check in self.checks:
                checks_str += f"{indent * 2}{check}\n"
            checks_str += f"{indent}]"
        else:
            checks_str = f"{indent}checks=[]"

        if self.parsers:
            parsers_str = f"{indent}parsers=[\n"
            for parser in self.parsers:
                parsers_str += f"{indent * 2}{parser}\n"
            parsers_str += f"{indent}]"
        else:
            parsers_str = f"{indent}parsers=[]"

        # add additional indents
        index_ = str(self.index).split("\n")
        if len(index_) == 1:
            index = str(self.index)
        else:
            index = "\n".join(
                x if i == 0 else f"{indent}{x}" for i, x in enumerate(index_)
            )

        return (
            f"<Schema {self.__class__.__name__}(\n"
            f"{columns_str},\n"
            f"{checks_str},\n"
            f"{parsers_str},\n"
            f"{indent}coerce={self.coerce},\n"
            f"{indent}dtype={self._dtype},\n"
            f"{indent}index={index},\n"
            f"{indent}strict={self.strict},\n"
            f"{indent}name={self.name},\n"
            f"{indent}ordered={self.ordered},\n"
            f"{indent}unique_column_names={self.unique_column_names},\n"
            f"{indent}metadata={self.metadata}, \n"
            f"{indent}add_missing_columns={self.add_missing_columns}\n"
            ")>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        def _compare_dict(obj):
            return {
                k: v for k, v in obj.__dict__.items() if k != "_IS_INFERRED"
            }

        return _compare_dict(self) == _compare_dict(other)

    if PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls._pydantic_validate,
            )

    else:

        @classmethod
        def __get_validators__(cls):
            yield cls._pydantic_validate

    @classmethod
    def _pydantic_validate(cls, schema: Any) -> Self:
        """Verify that the input is a compatible DataFrameSchema."""
        if not isinstance(schema, cls):  # type: ignore
            raise TypeError(f"{schema} is not a {cls}.")

        return cast(Self, schema)

    #################################
    # Schema Transformation Methods #
    #################################

    @inferred_schema_guard
    def add_columns(self, extra_schema_cols: Dict[str, Any]) -> Self:
        """Create a copy of the :class:`DataFrameSchema` with extra columns.

        :param extra_schema_cols: Additional columns of the format
        :type extra_schema_cols: DataFrameSchema
        :returns: a new :class:`DataFrameSchema` with the extra_schema_cols
            added.

        :example:

        To add columns to the schema, pass a dictionary with column name and
        ``Column`` instance key-value pairs.

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema(
        ...    {
        ...        "category": pa.Column(str),
        ...        "probability": pa.Column(float),
        ...    }
        ... )
        >>> print(
        ...     example_schema.add_columns({"even_number": pa.Column(pa.Bool)})
        ... )
        <Schema DataFrameSchema(
            columns={
                'category': <Schema Column(name=category, type=DataType(str))>
                'probability': <Schema Column(name=probability, type=DataType(float64))>
                'even_number': <Schema Column(name=even_number, type=DataType(bool))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. seealso:: :func:`remove_columns`

        """
        schema_copy = copy.deepcopy(self)
        schema_copy.columns = {
            **schema_copy.columns,
            **self.__class__(extra_schema_cols).columns,
        }
        return cast(Self, schema_copy)

    @inferred_schema_guard
    def remove_columns(self, cols_to_remove: List[str]) -> Self:
        """Removes columns from a :class:`DataFrameSchema` and returns a new
        copy.

        :param cols_to_remove: Columns to be removed from the
            ``DataFrameSchema``
        :type cols_to_remove: List
        :returns: a new :class:`DataFrameSchema` without the cols_to_remove
        :raises: :class:`~pandera.errors.SchemaInitError`: if column not in
            schema.

        :example:

        To remove a column or set of columns from a schema, pass a list of
        columns to be removed:

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema(
        ...     {
        ...         "category" : pa.Column(str),
        ...         "probability": pa.Column(float)
        ...     }
        ... )
        >>>
        >>> print(example_schema.remove_columns(["category"]))
        <Schema DataFrameSchema(
            columns={
                'probability': <Schema Column(name=probability, type=DataType(float64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. seealso:: :func:`add_columns`

        """
        schema_copy = copy.deepcopy(self)

        # ensure all specified keys are present in the columns
        not_in_cols: List[str] = [
            x for x in cols_to_remove if x not in schema_copy.columns.keys()
        ]
        if not_in_cols:
            raise errors.SchemaInitError(
                f"Keys {not_in_cols} not found in schema columns!"
            )

        for col in cols_to_remove:
            schema_copy.columns.pop(col)

        return cast(Self, schema_copy)

    @inferred_schema_guard
    def update_column(self, column_name: str, **kwargs) -> Self:
        """Create copy of a :class:`DataFrameSchema` with updated column
        properties.

        :param column_name:
        :param kwargs: key-word arguments supplied to
            :class:`~pandera.api.pandas.components.Column`
        :returns: a new :class:`DataFrameSchema` with updated column
        :raises: :class:`~pandera.errors.SchemaInitError`: if column not in
            schema or you try to change the name.

        :example:

        Calling ``schema.1`` returns the :class:`DataFrameSchema`
        with the updated column.

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema({
        ...     "category" : pa.Column(str),
        ...     "probability": pa.Column(float)
        ... })
        >>> print(
        ...     example_schema.update_column(
        ...         'category', dtype=pa.Category
        ...     )
        ... )
        <Schema DataFrameSchema(
            columns={
                'category': <Schema Column(name=category, type=DataType(category))>
                'probability': <Schema Column(name=probability, type=DataType(float64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. seealso:: :func:`rename_columns`

        """
        # check that columns exist in schema

        schema = self
        if "name" in kwargs:
            raise ValueError("cannot update 'name' of the column.")
        if column_name not in schema.columns:
            raise ValueError(f"column '{column_name}' not in {schema}")
        schema_copy = copy.deepcopy(schema)
        column_copy = copy.deepcopy(schema.columns[column_name])
        new_column = column_copy.__class__(
            **{**column_copy.properties, **kwargs}
        )
        schema_copy.columns.update({column_name: new_column})
        return cast(Self, schema_copy)

    def update_columns(
        self,
        update_dict: Dict[str, Dict[str, Any]],
    ) -> Self:
        """
        Create copy of a :class:`DataFrameSchema` with updated column
        properties.

        :param update_dict:
        :return: a new :class:`DataFrameSchema` with updated columns
        :raises: :class:`~pandera.errors.SchemaInitError`: if column not in
            schema or you try to change the name.

        :example:

        Calling ``schema.update_columns`` returns the :class:`DataFrameSchema`
        with the updated columns.

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema({
        ...     "category" : pa.Column(str),
        ...     "probability": pa.Column(float)
        ... })
        >>>
        >>> print(
        ...     example_schema.update_columns(
        ...         {"category": {"dtype":pa.Category}}
        ...     )
        ... )
        <Schema DataFrameSchema(
            columns={
                'category': <Schema Column(name=category, type=DataType(category))>
                'probability': <Schema Column(name=probability, type=DataType(float64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        """
        # pylint: disable=import-outside-toplevel,import-outside-toplevel
        from pandera.api.pandas.components import Column

        new_schema = copy.deepcopy(self)

        # ensure all specified keys are present in the columns
        not_in_cols: List[str] = [
            x for x in update_dict.keys() if x not in new_schema.columns.keys()
        ]
        if not_in_cols:
            raise errors.SchemaInitError(
                f"Keys {not_in_cols} not found in schema columns!"
            )

        new_columns: Dict[str, Column] = {}
        for col in new_schema.columns:
            # check
            if update_dict.get(col):
                if update_dict[col].get("name"):
                    raise errors.SchemaInitError(
                        "cannot update 'name' \
                                             property of the column."
                    )
            original_properties = new_schema.columns[col].properties
            if update_dict.get(col):
                new_properties = copy.deepcopy(original_properties)
                new_properties.update(update_dict[col])
                new_columns[col] = new_schema.columns[col].__class__(
                    **new_properties
                )
            else:
                new_columns[col] = new_schema.columns[col].__class__(
                    **original_properties
                )

        new_schema.columns = new_columns

        return cast(Self, new_schema)

    def rename_columns(self, rename_dict: Dict[str, str]) -> Self:
        """Rename columns using a dictionary of key-value pairs.

        :param rename_dict: dictionary of 'old_name': 'new_name' key-value
            pairs.
        :returns: :class:`DataFrameSchema` (copy of original)
        :raises: :class:`~pandera.errors.SchemaInitError` if column not in the
            schema.

        :example:

        To rename a column or set of columns, pass a dictionary of old column
        names and new column names, similar to the pandas DataFrame method.

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema({
        ...     "category" : pa.Column(str),
        ...     "probability": pa.Column(float)
        ... })
        >>>
        >>> print(
        ...     example_schema.rename_columns({
        ...         "category": "categories",
        ...         "probability": "probabilities"
        ...     })
        ... )
        <Schema DataFrameSchema(
            columns={
                'categories': <Schema Column(name=categories, type=DataType(str))>
                'probabilities': <Schema Column(name=probabilities, type=DataType(float64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. seealso:: :func:`update_column`

        """
        new_schema = copy.deepcopy(self)

        # ensure all specified keys are present in the columns
        not_in_cols: List[str] = [
            x for x in rename_dict.keys() if x not in new_schema.columns.keys()
        ]
        if not_in_cols:
            raise errors.SchemaInitError(
                f"Keys {not_in_cols} not found in schema columns!"
            )

        # remove any mapping to itself as this is a no-op
        rename_dict = {k: v for k, v in rename_dict.items() if k != v}

        # ensure all new keys are not present in the current column names
        already_in_columns: List[str] = [
            x for x in rename_dict.values() if x in new_schema.columns.keys()
        ]
        if already_in_columns:
            raise errors.SchemaInitError(
                f"Keys {already_in_columns} already found in schema columns!"
            )

        # We iterate over the existing columns dict and replace those keys
        # that exist in the rename_dict

        new_columns = {
            (rename_dict[col_name] if col_name in rename_dict else col_name): (
                col_attrs.set_name(rename_dict[col_name])
                if col_name in rename_dict
                else col_attrs
            )
            for col_name, col_attrs in new_schema.columns.items()
        }

        new_schema.columns = new_columns
        return cast(Self, new_schema)

    def select_columns(self, columns: List[Any]) -> Self:
        """Select subset of columns in the schema.

        *New in version 0.4.5*

        :param columns: list of column names to select.
        :returns:  :class:`DataFrameSchema` (copy of original) with only
            the selected columns.
        :raises: :class:`~pandera.errors.SchemaInitError` if column not in the
            schema.

        :example:

        To subset a schema by column, and return a new schema:

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema({
        ...     "category" : pa.Column(str),
        ...     "probability": pa.Column(float)
        ... })
        >>>
        >>> print(example_schema.select_columns(['category']))
        <Schema DataFrameSchema(
            columns={
                'category': <Schema Column(name=category, type=DataType(str))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. note:: If an index is present in the schema, it will also be
            included in the new schema.

        """

        new_schema = copy.deepcopy(self)

        # ensure all specified keys are present in the columns
        not_in_cols: List[str] = [
            x for x in columns if x not in new_schema.columns.keys()
        ]
        if not_in_cols:
            raise errors.SchemaInitError(
                f"Keys {not_in_cols} not found in schema columns!"
            )

        new_columns = {
            col_name: column
            for col_name, column in self.columns.items()
            if col_name in columns
        }
        new_schema.columns = new_columns
        return cast(Self, new_schema)

    def set_index(
        self, keys: List[str], drop: bool = True, append: bool = False
    ) -> Self:
        """
        A method for setting the :class:`Index` of a :class:`DataFrameSchema`,
        via an existing :class:`Column` or list of columns.

        :param keys: list of labels
        :param drop: bool, default True
        :param append: bool, default False
        :return: a new :class:`DataFrameSchema` with specified column(s) in the
            index.
        :raises: :class:`~pandera.errors.SchemaInitError` if column not in the
            schema.

        :examples:

        Just as you would set the index in a ``pandas`` DataFrame from an
        existing column, you can set an index within the schema from an
        existing column in the schema.

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema({
        ...     "category" : pa.Column(str),
        ...     "probability": pa.Column(float)})
        >>>
        >>> print(example_schema.set_index(['category']))
        <Schema DataFrameSchema(
            columns={
                'probability': <Schema Column(name=probability, type=DataType(float64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=<Schema Index(name=category, type=DataType(str))>,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        If you have an existing index in your schema, and you would like to
        append a new column as an index to it (yielding a :class:`Multiindex`),
        just use set_index as you would in pandas.

        >>> example_schema = pa.DataFrameSchema(
        ...     {
        ...         "column1": pa.Column(str),
        ...         "column2": pa.Column(int)
        ...     },
        ...     index=pa.Index(name = "column3", dtype = int)
        ... )
        >>>
        >>> print(example_schema.set_index(["column2"], append = True))
        <Schema DataFrameSchema(
            columns={
                'column1': <Schema Column(name=column1, type=DataType(str))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=<Schema MultiIndex(
                indexes=[
                    <Schema Index(name=column3, type=DataType(int64))>
                    <Schema Index(name=column2, type=DataType(int64))>
                ]
                coerce=False,
                strict=False,
                name=None,
                ordered=True
            )>,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. seealso:: :func:`reset_index`

        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from pandera.api.pandas.components import Index, MultiIndex

        new_schema = copy.deepcopy(self)

        keys_temp: List = (
            list(set(keys)) if not isinstance(keys, list) else keys
        )

        # ensure all specified keys are present in the columns
        not_in_cols: List[str] = [
            x for x in keys_temp if x not in new_schema.columns.keys()
        ]
        if not_in_cols:
            raise errors.SchemaInitError(
                f"Keys {not_in_cols} not found in schema columns!"
            )

        # if there is already an index, append or replace according to
        # parameters
        ind_list: List = (
            []
            if new_schema.index is None or not append
            else (
                list(new_schema.index.indexes)
                if isinstance(new_schema.index, MultiIndex) and append
                else [new_schema.index]
            )
        )

        for col in keys_temp:
            ind_list.append(
                Index(
                    dtype=new_schema.columns[col].dtype,
                    name=col,
                    checks=new_schema.columns[col].checks,
                    nullable=new_schema.columns[col].nullable,
                    unique=new_schema.columns[col].unique,
                    coerce=new_schema.columns[col].coerce,
                )
            )

        new_schema.index = (
            ind_list[0] if len(ind_list) == 1 else MultiIndex(ind_list)
        )

        # if drop is True as defaulted, drop the columns moved into the index
        if drop:
            new_schema = new_schema.remove_columns(keys_temp)

        return cast(Self, new_schema)

    def reset_index(
        self, level: Optional[List[str]] = None, drop: bool = False
    ) -> Self:
        """
        A method for resetting the :class:`Index` of a :class:`DataFrameSchema`

        :param level: list of labels
        :param drop: bool, default True
        :return: a new :class:`DataFrameSchema` with specified column(s) in the
            index.
        :raises: :class:`~pandera.errors.SchemaInitError` if no index set in
            schema.
        :examples:

        Similar to the ``pandas`` reset_index method on a pandas DataFrame,
        this method can be used to to fully or partially reset indices of a
        schema.

        To remove the entire index from the schema, just call the reset_index
        method with default parameters.

        >>> import pandera as pa
        >>>
        >>> example_schema = pa.DataFrameSchema(
        ...     {"probability" : pa.Column(float)},
        ...     index = pa.Index(name="unique_id", dtype=int)
        ... )
        >>>
        >>> print(example_schema.reset_index())
        <Schema DataFrameSchema(
            columns={
                'probability': <Schema Column(name=probability, type=DataType(float64))>
                'unique_id': <Schema Column(name=unique_id, type=DataType(int64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=None,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        This reclassifies an index (or indices) as a column (or columns).

        Similarly, to partially alter the index, pass the name of the column
        you would like to be removed to the ``level`` parameter, and you may
        also decide whether to drop the levels with the ``drop`` parameter.

        >>> example_schema = pa.DataFrameSchema({
        ...     "category" : pa.Column(str)},
        ...     index = pa.MultiIndex([
        ...         pa.Index(name="unique_id1", dtype=int),
        ...         pa.Index(name="unique_id2", dtype=str)
        ...         ]
        ...     )
        ... )
        >>> print(example_schema.reset_index(level = ["unique_id1"]))
        <Schema DataFrameSchema(
            columns={
                'category': <Schema Column(name=category, type=DataType(str))>
                'unique_id1': <Schema Column(name=unique_id1, type=DataType(int64))>
            },
            checks=[],
            parsers=[],
            coerce=False,
            dtype=None,
            index=<Schema Index(name=unique_id2, type=DataType(str))>,
            strict=False,
            name=None,
            ordered=False,
            unique_column_names=False,
            metadata=None,
            add_missing_columns=False
        )>

        .. seealso:: :func:`set_index`

        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from pandera.api.pandas.components import Column, Index, MultiIndex

        # explcit check for an empty list
        if level == []:
            return self

        new_schema = copy.deepcopy(self)

        if new_schema.index is None:
            raise errors.SchemaInitError(
                "There is currently no index set for this schema."
            )

        # ensure no duplicates
        level_temp: Union[List[Any], List[str]] = (
            new_schema.index.names if level is None else list(set(level))
        )

        # ensure all specified keys are present in the index
        level_not_in_index: Union[List[Any], List[str], None] = (
            [x for x in level_temp if x not in new_schema.index.names]
            if isinstance(new_schema.index, MultiIndex) and level_temp
            else (
                []
                if isinstance(new_schema.index, Index)
                and (level_temp == [new_schema.index.name])
                else level_temp
            )
        )
        if level_not_in_index:
            raise errors.SchemaInitError(
                f"Keys {level_not_in_index} not found in schema columns!"
            )

        new_index = (
            None
            if not level_temp or isinstance(new_schema.index, Index)
            else new_schema.index.remove_columns(level_temp)
        )
        new_index = (
            new_index
            if new_index is None
            else (
                Index(
                    dtype=new_index.columns[list(new_index.columns)[0]].dtype,
                    checks=new_index.columns[
                        list(new_index.columns)[0]
                    ].checks,
                    nullable=new_index.columns[
                        list(new_index.columns)[0]
                    ].nullable,
                    unique=new_index.columns[
                        list(new_index.columns)[0]
                    ].unique,
                    coerce=new_index.columns[
                        list(new_index.columns)[0]
                    ].coerce,
                    name=new_index.columns[list(new_index.columns)[0]].name,
                )
                if (len(list(new_index.columns)) == 1)
                and (new_index is not None)
                else (
                    None
                    if (len(list(new_index.columns)) == 0)
                    and (new_index is not None)
                    else new_index
                )
            )
        )

        if not drop:
            additional_columns: Dict[str, Any] = (
                {col: new_schema.index.columns.get(col) for col in level_temp}
                if isinstance(new_schema.index, MultiIndex)
                else {new_schema.index.name: new_schema.index}
            )
            new_schema = new_schema.add_columns(
                {
                    k: Column(
                        dtype=v.dtype,
                        parsers=v.parsers,
                        checks=v.checks,
                        nullable=v.nullable,
                        unique=v.unique,
                        coerce=v.coerce,
                        name=v.name,
                    )
                    for (k, v) in additional_columns.items()
                }
            )

        new_schema.index = new_index

        return new_schema

    #####################
    # Schema IO Methods #
    #####################

    def to_script(self, fp: Optional[Union[str, Path]] = None) -> Self:
        """Write DataFrameSchema to python script.

        :param path: str, Path to write script
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.to_script(self, fp)

    @classmethod
    def from_yaml(cls, yaml_schema) -> Self:
        """Create DataFrameSchema from yaml file.

        :param yaml_schema: str, Path to yaml schema, or serialized yaml
            string.
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.from_yaml(yaml_schema)

    def to_yaml(self, stream: Optional[os.PathLike] = None) -> Optional[str]:
        """Write DataFrameSchema to yaml file.

        :param stream: file stream to write to. If None, dumps to string.
        :returns: yaml string if stream is None, otherwise returns None.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.to_yaml(self, stream=stream)

    @classmethod
    def from_json(cls, source) -> Self:
        """Create DataFrameSchema from json file.

        :param source: str, Path to json schema, or serialized yaml
            string.
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.from_json(source)

    @overload
    def to_json(
        self, target: None = None, **kwargs
    ) -> str:  # pragma: no cover
        ...

    @overload
    def to_json(
        self, target: os.PathLike, **kwargs
    ) -> None:  # pragma: no cover
        ...

    def to_json(
        self, target: Optional[os.PathLike] = None, **kwargs
    ) -> Optional[str]:
        """Write DataFrameSchema to json file.

        :param target: file target to write to. If None, dumps to string.
        :returns: json string if target is None, otherwise returns None.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.to_json(self, target, **kwargs)

    ###########################
    # Schema Strategy Methods #
    ###########################

    @st.strategy_import_error
    def strategy(
        self, *, size: Optional[int] = None, n_regex_columns: int = 1
    ):
        """Create a ``hypothesis`` strategy for generating a DataFrame.

        :param size: number of elements to generate
        :param n_regex_columns: number of regex columns to generate.
        :returns: a strategy that generates pandas DataFrame objects.
        """
        return st.dataframe_strategy(
            self.dtype,
            columns=self.columns,
            checks=self.checks,
            unique=self.unique,
            index=self.index,
            size=size,
            n_regex_columns=n_regex_columns,
        )

    def example(
        self, size: Optional[int] = None, n_regex_columns: int = 1
    ) -> TDataObject:
        """Generate an example of a particular size.

        :param size: number of elements in the generated DataFrame.
        :returns: pandas DataFrame object.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(
                size=size, n_regex_columns=n_regex_columns
            ).example()


def _validate_columns(
    column_dict: dict[Any, Any],  # type: ignore [name-defined]
) -> None:
    for column_name, column in column_dict.items():
        for check in column.checks:
            if check.groupby is None or callable(check.groupby):
                continue
            nonexistent_groupby_columns = [
                c for c in check.groupby if c not in column_dict
            ]
            if nonexistent_groupby_columns:
                raise errors.SchemaInitError(
                    f"groupby argument {nonexistent_groupby_columns} in "
                    f"Check for Column {column_name} not "
                    "specified in the DataFrameSchema."
                )


def _columns_renamed(
    columns: dict[Any, Any],  # type: ignore [name-defined]
) -> dict[Any, Any]:  # type: ignore [name-defined]
    def renamed(column, new_name):
        column = copy.deepcopy(column)
        column.set_name(new_name)
        return column

    return {
        column_name: renamed(column, column_name)
        for column_name, column in columns.items()
    }
