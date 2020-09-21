"""Core pandera schema class definitions."""

import json
import copy
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict, Any

import pandas as pd

from . import errors, constants, dtypes
from .checks import Check
from .dtypes import PandasDtype, PandasExtensionType
from .error_formatters import (
    format_generic_error_message, format_vectorized_error_message,
    reshape_failure_cases, scalar_failure_case,
)
from .error_handlers import SchemaErrorHandler
from .hypotheses import Hypothesis


N_INDENT_SPACES = 4

CheckList = Optional[
    Union[
        Union[Check, Hypothesis],
        List[Union[Check, Hypothesis]]
    ]
]

PandasDtypeInputTypes = Union[str, type, PandasDtype, PandasExtensionType]


def _inferred_schema_guard(method):
    """
    Invoking a method wrapped with this decorator will set _is_inferred to
    False.
    """

    # pylint: disable=inconsistent-return-statements
    @wraps(method)
    def _wrapper(schema, *args, **kwargs):
        new_schema = method(schema, *args, **kwargs)
        if new_schema is not None and id(new_schema) != id(schema):
            # if method returns a copy of the schema object,
            # the original schema instance and the copy should be set to
            # not inferred.
            new_schema._is_inferred = False  # pylint: disable=protected-access
            return new_schema
        schema._is_inferred = False  # pylint: disable=protected-access

    return _wrapper


class DataFrameSchema():
    """A light-weight pandas DataFrame validator."""

    def __init__(
            self,
            columns: Dict[Any, Any] = None,
            checks: CheckList = None,
            index=None,
            transformer: Callable = None,
            coerce: bool = False,
            strict=False,
            name: str = None) -> None:
        """Initialize DataFrameSchema validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :type columns: mapping of column names and column schema component.
        :param checks: dataframe-wide checks.
        :param index: specify the datatypes and properties of the index.
        :param transformer: a callable with signature:
            pandas.DataFrame -> pandas.DataFrame. If specified, calling
            `validate` will verify properties of the columns and return the
            transformed dataframe object.
        :param coerce: whether or not to coerce all of the columns on
            validation.
        :param strict: whether or not to accept columns in the dataframe that
            aren't in the DataFrameSchema.
        :param name: name of the schema.

        :raises SchemaInitError: if impossible to build schema from parameters

        :examples:

        >>> import pandera as pa
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "str_column": pa.Column(pa.String),
        ...     "float_column": pa.Column(pa.Float),
        ...     "int_column": pa.Column(pa.Int),
        ...     "date_column": pa.Column(pa.DateTime),
        ... })

        Use the pandas API to define checks, which takes a function with
        the signature: ``pd.Series -> Union[bool, pd.Series]`` where the
        output series contains boolean values.

        >>> from pandera import Check
        >>>
        >>> schema_withchecks = pa.DataFrameSchema({
        ...     "probability": pa.Column(
        ...         pa.Float, pa.Check(lambda s: (s >= 0) & (s <= 1))),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": pa.Column(
        ...         pa.String, [
        ...             pa.Check(lambda s: s.isin(["dog", "cat", "duck"])),
        ...             pa.Check(lambda s: (s == "dog").mean() > 0.5),
        ...         ]),
        ... })

        See :ref:`here<DataFrameSchemas>` for more usage details.

        """
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        self.columns = {} if columns is None else columns

        if coerce:
            missing_pandas_type = [
                name for name, col in self.columns.items()
                if col.pandas_dtype is None
            ]
            if missing_pandas_type:
                raise errors.SchemaInitError(
                    "Must specify dtype in all Columns if coercing "
                    "DataFrameSchema ; columns with missing pandas_type:" +
                    ", ".join(missing_pandas_type))

        self.checks = checks
        self.index = index
        self.transformer = transformer
        self.strict = strict
        self.name = name
        self._coerce = coerce
        self._validate_schema()
        self._set_column_names()

        # this attribute is not meant to be accessed by users and is explicitly
        # set to True in the case that a schema is created by infer_schema.
        self._IS_INFERRED = False

    @property
    def coerce(self):
        """Whether to coerce series to specified type."""
        return self._coerce

    # the _is_inferred getter and setter methods are not public
    @property
    def _is_inferred(self):
        return self._IS_INFERRED

    @_is_inferred.setter
    def _is_inferred(self, value: bool):
        self._IS_INFERRED = value

    def _validate_schema(self):
        for column_name, column in self.columns.items():
            for check in column.checks:
                if check.groupby is None or callable(check.groupby):
                    continue
                nonexistent_groupby_columns = [
                    c for c in check.groupby if c not in self.columns]
                if nonexistent_groupby_columns:
                    raise errors.SchemaInitError(
                        "groupby argument %s in Check for Column %s not "
                        "specified in the DataFrameSchema." %
                        (nonexistent_groupby_columns, column_name))

    def _set_column_names(self):

        def _set_column_handler(column, column_name):
            if column.name is not None and column.name != column_name:
                warnings.warn(
                    "resetting column for %s to '%s'." % (column, column_name))
            elif column.name == column_name:
                return column
            return column.set_name(column_name)

        self.columns = {
            column_name: _set_column_handler(column, column_name)
            for column_name, column in self.columns.items()
        }

    @staticmethod
    def _dataframe_to_validate(
            dataframe: pd.DataFrame,
            head: Optional[int],
            tail: Optional[int],
            sample: Optional[int],
            random_state: Optional[int]) -> pd.DataFrame:
        dataframe_subsample = []
        if head is not None:
            dataframe_subsample.append(dataframe.head(head))
        if tail is not None:
            dataframe_subsample.append(dataframe.tail(tail))
        if sample is not None:
            dataframe_subsample.append(
                dataframe.sample(sample, random_state=random_state))
        return dataframe if not dataframe_subsample else \
            pd.concat(dataframe_subsample).drop_duplicates()

    @property
    def dtype(self) -> Dict[str, str]:
        """
        A pandas style dtype dict where the keys are column names and values
        are pandas dtype for the column. Excludes columns where regex=True.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_columns = [
            name for name, col in self.columns.items() if col.regex]
        if regex_columns:
            warnings.warn(
                "Schema has columns specified as regex column names: %s "
                "Use the `get_dtype` to get the datatypes for these "
                "columns." % regex_columns,
                UserWarning
            )
        return {n: c.dtype for n, c in self.columns.items() if not c.regex}

    def get_dtype(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """
        Same as the ``dtype`` property, but expands columns where
        ``regex == True`` based on the supplied dataframe.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_dtype = {}
        for _, column in self.columns.items():
            if column.regex:
                regex_dtype.update({
                    c: column.dtype for c in
                    column.get_regex_columns(dataframe.columns)
                })
        return {
            **{n: c.dtype for n, c in self.columns.items() if not c.regex},
            **regex_dtype,
        }

    def validate(
            self,
            check_obj: pd.DataFrame,
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> pd.DataFrame:
        # pylint: disable=duplicate-code,too-many-locals
        """Check if all columns in a dataframe have a column in the Schema.

        :param pd.DataFrame dataframe: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrorReport``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :returns: validated ``DataFrame``

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        Calling ``schema.validate`` returns the dataframe.

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> df = pd.DataFrame({
        ...     "probability": [0.1, 0.4, 0.52, 0.23, 0.8, 0.76],
        ...     "category": ["dog", "dog", "cat", "duck", "dog", "dog"]
        ... })
        >>>
        >>> schema_withchecks = pa.DataFrameSchema({
        ...     "probability": pa.Column(
        ...         pa.Float, pa.Check(lambda s: (s >= 0) & (s <= 1))),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": pa.Column(
        ...         pa.String, [
        ...             pa.Check(lambda s: s.isin(["dog", "cat", "duck"])),
        ...             pa.Check(lambda s: (s == "dog").mean() > 0.5),
        ...         ]),
        ... })
        >>>
        >>> schema_withchecks.validate(df)[["probability", "category"]]
           probability category
        0         0.10      dog
        1         0.40      dog
        2         0.52      cat
        3         0.23     duck
        4         0.80      dog
        5         0.76      dog
        """

        # pylint: disable=too-many-branches
        if self._is_inferred:
            warnings.warn(
                "This %s is an inferred schema that hasn't been "
                "modified. It's recommended that you refine the schema "
                "by calling `add_columns`, `remove_columns`, or "
                "`update_columns` before using it to validate data."
                % type(self),
                UserWarning
            )

        error_handler = SchemaErrorHandler(lazy)

        # dataframe strictness check makes sure all columns in the dataframe
        # are specified in the dataframe schema
        if self.strict:

            # expand regex columns
            col_regex_matches = []  # type: ignore
            for colname, col_schema in self.columns.items():
                if col_schema.regex:
                    try:
                        col_regex_matches.extend(
                            col_schema.get_regex_columns(check_obj.columns))
                    except errors.SchemaError:
                        pass

            expanded_column_names = frozenset(
                [n for n, c in self.columns.items() if not c.regex] +
                col_regex_matches
            )

            for column in check_obj:
                if column not in expanded_column_names:
                    msg = (
                        "column '%s' not in DataFrameSchema %s" %
                        (column, self.columns)
                    )
                    error_handler.collect_error(
                        "column_not_in_schema", errors.SchemaError(
                            self, check_obj, msg,
                            failure_cases=scalar_failure_case(column),
                            check="column_in_schema",
                        )
                    )

        # column data-type coercion logic
        lazy_exclude_columns = []
        for colname, col_schema in self.columns.items():
            if col_schema.regex:
                try:
                    matched_columns = col_schema.get_regex_columns(
                        check_obj.columns)
                except errors.SchemaError:
                    matched_columns = pd.Index([])

                for matched_colname in matched_columns:
                    if col_schema.coerce or self.coerce:
                        check_obj[matched_colname] = col_schema.coerce_dtype(
                            check_obj[matched_colname])

            elif colname not in check_obj and col_schema.required:
                if lazy:
                    # exclude columns that are not present in the dataframe
                    # for lazy validation, the error is collected by the
                    # error_handler and should raise a SchemaErrors exception
                    # at the end of the `validate` method.
                    lazy_exclude_columns.append(colname)
                msg = (
                    "column '%s' not in dataframe\n%s" %
                    (colname, check_obj.head())
                )
                error_handler.collect_error(
                    "column_not_in_dataframe", errors.SchemaError(
                        self, check_obj, msg,
                        failure_cases=scalar_failure_case(colname),
                        check="column_in_dataframe",
                    )
                )

            elif col_schema.coerce or self.coerce:
                check_obj.loc[:, colname] = col_schema.coerce_dtype(
                    check_obj[colname])

        schema_components = [
            col for col_name, col in self.columns.items()
            if (col.required or col_name in check_obj)
            and col_name not in lazy_exclude_columns
        ]
        if self.index is not None:
            if self.index.coerce or self.coerce:
                check_obj.index = self.index.coerce_dtype(check_obj.index)
            schema_components.append(self.index)

        dataframe_to_validate = self._dataframe_to_validate(
            check_obj, head, tail, sample, random_state)

        check_results = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                check_results.append(isinstance(
                    schema_component(dataframe_to_validate), pd.DataFrame))
            except errors.SchemaError as err:
                error_handler.collect_error("schema_component_check", err)

        # dataframe-level checks
        for check_index, check in enumerate(self.checks):
            try:
                check_results.append(_handle_check_results(
                    self, check_index, check, dataframe_to_validate))
            except errors.SchemaError as err:
                error_handler.collect_error("dataframe_check", err)

        if lazy and error_handler.collected_errors:
            raise errors.SchemaErrors(
                error_handler.collected_errors, check_obj)

        assert all(check_results)

        if self.transformer is not None:
            check_obj = self.transformer(check_obj)

        return check_obj

    def __call__(
            self,
            dataframe: pd.DataFrame,
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False):
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
            checks and raises a ``SchemaErrorReport``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        """
        return self.validate(dataframe, head, tail, sample, random_state, lazy)

    def __repr__(self):
        """Represent string for logging."""
        return "%s(columns=%s, index=%s, transformer=%s, coerce=%s)" % \
            (self.__class__.__name__,
             self.columns,
             self.index,
             self.transformer,
             self.coerce)

    def __str__(self):
        """Represent string for user inspection."""

        def _format_multiline(json_str, arg):
            return "\n".join(
                "{}{}".format(_indent, line) if i != 0
                else "{}{}={}".format(_indent, arg, line)
                for i, line in enumerate(json_str.split("\n"))
            )

        columns = {k: str(v) for k, v in self.columns.items()}
        columns = json.dumps(columns, indent=N_INDENT_SPACES)
        _indent = " " * N_INDENT_SPACES
        columns = _format_multiline(columns, "columns")
        checks = None if self.checks is None else _format_multiline(
            json.dumps([str(x) for x in self.checks], indent=N_INDENT_SPACES),
            "checks")
        return (
            "{class_name}(\n"
            "{columns},\n"
            "{checks},\n"
            "{indent}index={index},\n"
            "{indent}transformer={transformer},\n"
            "{indent}coerce={coerce},\n"
            "{indent}strict={strict}\n"
            ")"
        ).format(
            class_name=self.__class__.__name__,
            columns=columns,
            checks=checks,
            index=str(self.index),
            transformer=str(self.transformer),
            coerce=self.coerce,
            strict=self.strict,
            indent=_indent,
        )

    def __eq__(self, other):
        def _compare_dict(obj):
            return {
                k: v for k, v in obj.__dict__.items()
                if k != "_IS_INFERRED"
            }
        # if _compare_dict(self) != _compare_dict(other):
        #     import ipdb; ipdb.set_trace()
        return _compare_dict(self) == _compare_dict(other)

    @_inferred_schema_guard
    def add_columns(self,
                    extra_schema_cols: Dict[str, Any]) -> 'DataFrameSchema':
        """Create a copy of the DataFrameSchema with extra columns.

        :param extra_schema_cols: Additional columns of the format
        :type extra_schema_cols: DataFrameSchema
        :returns: a new DataFrameSchema with the extra_schema_cols added

        """
        schema_copy = copy.deepcopy(self)
        schema_copy.columns = {
            **schema_copy.columns,
            **DataFrameSchema(extra_schema_cols).columns
        }
        return schema_copy

    @_inferred_schema_guard
    def remove_columns(self,
                       cols_to_remove: List) -> 'DataFrameSchema':
        """Removes columns from a DataFrameSchema and returns a new copy.

        :param cols_to_remove: Columns to be removed from the DataFrameSchema
        :type cols_to_remove: List
        :returns: a new DataFrameSchema without the cols_to_remove

        """
        schema_copy = copy.deepcopy(self)
        for col in cols_to_remove:
            schema_copy.columns.pop(col)

        return schema_copy

    @_inferred_schema_guard
    def update_column(self, column_name: str, **kwargs) -> "DataFrameSchema":
        """Create copy of a DataFrameSchema with updated column properties.

        :param column_name:
        :param kwargs: key-word arguments supplied to :py:class:`Column`
        :returns: a new DataFrameSchema with updated column
        """
        if "name" in kwargs:
            raise ValueError("cannot update 'name' of the column.")
        if column_name not in self.columns:
            raise ValueError("column '%s' not in %s" % (column_name, self))
        schema_copy = copy.deepcopy(self)
        column_copy = copy.deepcopy(self.columns[column_name])
        new_column = column_copy.__class__(**{
            **column_copy.properties, **kwargs
        })
        schema_copy.columns.update({column_name: new_column})
        return schema_copy

    def rename_columns(self, rename_dict: dict):
        """Rename columns using a dictionary of key-value pairs.

        :param rename_dict: dictionary of 'old_name': 'new_name' key-value
            pairs.
        :returns: dataframe schema (copy of original)
        """

        # We iterate over the existing columns dict and replace those keys
        # that exist in the rename_dict
        new_schema = copy.deepcopy(self)
        new_columns = {
            (
                rename_dict[col_name]if col_name in rename_dict else col_name
            ): col_attrs
            for col_name, col_attrs in self.columns.items()
        }

        new_schema.columns = new_columns
        return new_schema

    def select_columns(self, columns: list):
        """Select subset of columns in the schema.

        *New in version 0.4.5*

        :param columns: list of column names to select.
        :returns: dataframe schema (copy of original)
        """
        new_schema = copy.deepcopy(self)
        new_columns = {
            col_name: column for col_name, column in self.columns.items()
            if col_name in columns
        }
        new_schema.columns = new_columns
        return new_schema

    @classmethod
    def from_yaml(cls, yaml_schema) -> "DataFrameSchema":
        """Create DataFrameSchema from yaml file.

        :param yaml_schema: str, Path to yaml schema, or serialized yaml
            string.
        :returns: dataframe schema.
        """
        import pandera.io  # pylint: disable-all
        return pandera.io.from_yaml(yaml_schema)

    def to_yaml(self, fp: Union[str, Path] = None):
        """Write DataFrameSchema to yaml file.

        :param dataframe_schema: schema to write to file or dump to string.
        :param stream: file stream to write to. If None, dumps to string.
        :returns: yaml string if stream is None, otherwise returns None.
        """
        import pandera.io  # pylint: disable-all
        return pandera.io.to_yaml(self, fp)


class SeriesSchemaBase():
    """Base series validator object."""

    def __init__(
            self,
            pandas_dtype: PandasDtypeInputTypes = None,
            checks: CheckList = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            name: str = None) -> None:
        """Initialize series schema base object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a
            pandas.Series object.
        :type checks: callable
        :param nullable: Whether or not column can contain null values.
        :type nullable: bool
        :param allow_duplicates:
        :type allow_duplicates: bool
        """
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        self._pandas_dtype = pandas_dtype
        self._nullable = nullable
        self._allow_duplicates = allow_duplicates
        self._coerce = coerce
        self._checks = checks
        self._name = name

        for check in self.checks:
            if check.groupby is not None and not self._allow_groupby:
                raise errors.SchemaInitError(
                    "Cannot use groupby checks with type %s" % type(self))

        # make sure pandas dtype is valid
        try:
            self.dtype
        except TypeError:
            raise

        # this attribute is not meant to be accessed by users and is explicitly
        # set to True in the case that a schema is created by infer_schema.
        self._IS_INFERRED = False

    # the _is_inferred getter and setter methods are not public
    @property
    def _is_inferred(self):
        return self._IS_INFERRED

    @_is_inferred.setter
    def _is_inferred(self, value: bool):
        self._IS_INFERRED = value

    @property
    def checks(self):
        """Return list of checks or hypotheses."""
        if self._checks is None:
            return []
        if isinstance(self._checks, (Check, Hypothesis)):
            return [self._checks]
        return self._checks

    @checks.setter
    def checks(self, checks):
        self._checks = checks

    @_inferred_schema_guard
    def set_checks(self, checks: CheckList):
        """Create a new SeriesSchema with a new set of Checks

        :param checks: checks to set on the new schema
        :returns: a new SeriesSchema with a new set of checks
        """
        schema_copy = copy.deepcopy(self)
        schema_copy.checks = checks
        return schema_copy

    @property
    def nullable(self) -> bool:
        """Whether the series is nullable."""
        return self._nullable

    @property
    def allow_duplicates(self) -> bool:
        """Whether to allow duplicate values."""
        return self._allow_duplicates

    @property
    def coerce(self) -> bool:
        """Whether to coerce series to specified type."""
        return self._coerce

    @property
    def name(self) -> Union[str, None]:
        """Get SeriesSchema name."""
        return self._name

    @property
    def pandas_dtype(self) -> Union[
            str,
            dtypes.PandasDtype,
            dtypes.PandasExtensionType]:
        """Get the pandas dtype"""
        return self._pandas_dtype

    @pandas_dtype.setter
    def pandas_dtype(self, value: Union[
            str,
            dtypes.PandasDtype,
            dtypes.PandasExtensionType]) -> None:
        """Set the pandas dtype"""
        self._pandas_dtype = value
        try:
            self.dtype
        except TypeError:
            raise

    @property
    def dtype(self) -> Union[str, None]:
        """String representation of the dtype."""
        try:
            is_extension_type = isinstance(
                self._pandas_dtype,
                pd.core.dtypes.base.ExtensionDtype)
        except (AttributeError, TypeError):
            is_extension_type = False

        if is_extension_type:
            dtype = str(self._pandas_dtype)
        elif self._pandas_dtype is None:
            dtype = self._pandas_dtype  # type: ignore
        elif isinstance(self._pandas_dtype, str):
            dtype = PandasDtype.from_str_alias(  # type: ignore
                self._pandas_dtype).str_alias
        elif isinstance(self._pandas_dtype, type):
            dtype = PandasDtype.from_python_type(self._pandas_dtype).str_alias
        elif isinstance(self._pandas_dtype, dtypes.PandasDtype):
            dtype = self._pandas_dtype.str_alias
        else:
            raise TypeError(
                "type of `pandas_dtype` argument not recognized: %s "
                "Please specify a pandera PandasDtype enum, legal pandas data "
                "type, pandas data type string alias, or numpy data type "
                "string alias" % type(self._pandas_dtype)
            )
        return dtype

    def coerce_dtype(
            self, series_or_index: Union[pd.Series, pd.Index]) -> pd.Series:
        """Coerce type of a pd.Series by type specified in pandas_dtype.

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).
        :returns: ``Series`` with coerced data type
        """
        if self._pandas_dtype is dtypes.PandasDtype.String:
            # only coerce non-null elements to string
            return series_or_index.where(
                series_or_index.isna(), series_or_index.astype(str))

        try:
            return series_or_index.astype(self.dtype)
        except TypeError as exc:
            msg = "Error while coercing '%s' to type %s" % (
                self.name, self.dtype
            )
            raise TypeError(msg) from exc

    @property
    def _allow_groupby(self):
        """Whether the schema or schema component allows groupby operations."""
        raise NotImplementedError(
            "The _allow_groupby property must be implemented by subclasses "
            "of SeriesSchemaBase")

    def validate(
            self,
            check_obj: Union[pd.DataFrame, pd.Series],
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        # pylint: disable=too-many-branches,W0212,too-many-locals,duplicate-code  # noqa
        """Validate a series or specific column in dataframe.

        :check_obj: pandas DataFrame or Series to validate.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrorReport``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :returns: validated DataFrame or Series.

        """

        if self._is_inferred:
            warnings.warn(
                "This %s is an inferred schema that hasn't been "
                "modified. It's recommended that you refine the schema "
                "by calling `set_checks` before using it to validate data." %
                type(self),
                UserWarning
            )

        error_handler = SchemaErrorHandler(lazy)

        check_obj_to_validate = _pandas_obj_to_validate(
            check_obj, head, tail, sample, random_state)

        series = check_obj_to_validate.copy() if \
            isinstance(check_obj_to_validate, pd.Series) \
            else check_obj_to_validate[self.name].copy()

        if series.name != self._name:
            msg = "Expected %s to have name '%s', found '%s'" % \
                (type(self), self._name, series.name)
            raise errors.SchemaError(
                self, check_obj, msg,
                failure_cases=scalar_failure_case(series.name),
                check="column_name('%s')" % self._name,
            )

        series_dtype = series.dtype
        if self._nullable:
            series_no_nans = series.dropna()
            if self.dtype in dtypes.NUMPY_NONNULLABLE_INT_DTYPES:
                _series = series_no_nans.astype(self.dtype)
                series_dtype = _series.dtype
                if (_series != series_no_nans).any():
                    # in case where dtype is meant to be int, make sure that
                    # casting to int results in equal values.
                    msg = (
                        "after dropping null values, expected values in "
                        "series '%s' to be int, found: %s" %
                        (series.name, set(series))
                    )
                    error_handler.collect_error(
                        "unexpected_nullable_integer_type",
                        errors.SchemaError(
                            self, check_obj, msg,
                            failure_cases=reshape_failure_cases(
                                series_no_nans
                            ),
                            check="nullable_integer",
                        )
                    )
        else:
            nulls = series.isna()
            if sum(nulls) > 0:
                msg = (
                    "non-nullable series '%s' contains null values: %s" %
                    (series.name,
                     series[nulls].head(
                        constants.N_FAILURE_CASES).to_dict())
                )
                error_handler.collect_error(
                    "series_contains_nulls",
                    errors.SchemaError(
                        self, check_obj, msg,
                        failure_cases=reshape_failure_cases(
                            series[nulls], ignore_na=False
                        ),
                        check="not_nullable",
                    )
                )

        # Check if the series contains duplicate values
        if not self._allow_duplicates:
            duplicates = series.duplicated()
            if any(duplicates):
                msg = (
                    "series '%s' contains duplicate values: %s" %
                    (series.name,
                     series[duplicates].head(
                         constants.N_FAILURE_CASES).to_dict())
                )
                error_handler.collect_error(
                    "series_contains_duplicates",
                    errors.SchemaError(
                        self, check_obj, msg,
                        failure_cases=reshape_failure_cases(
                            series[duplicates]
                        ),
                        check="no_duplicates",
                    )
                )

        if self.dtype is not None and str(series_dtype) != self.dtype:
            msg = (
                "expected series '%s' to have type %s, got %s" %
                (series.name, self.dtype, str(series_dtype))
            )
            error_handler.collect_error(
                "wrong_pandas_dtype",
                errors.SchemaError(
                    self, check_obj, msg,
                    failure_cases=scalar_failure_case(str(series_dtype)),
                    check="pandas_dtype('%s')" % self.dtype,
                )
            )

        if not self.checks:
            return check_obj

        check_results = []
        if isinstance(check_obj, pd.Series):
            check_obj, check_args = series, [None]
        else:
            check_obj = check_obj.loc[series.index.unique()].copy()
            check_args = [self.name]  # type: ignore

        for check_index, check in enumerate(self.checks):
            try:
                check_results.append(
                    _handle_check_results(
                        self, check_index, check, check_obj, *check_args
                    )
                )
            except errors.SchemaError as err:
                error_handler.collect_error("dataframe_check", err)
            except Exception as err:
                # catch other exceptions that may occur when executing the
                # Check
                err_str = '%s("%s")' % (err.__class__.__name__, err.args[0])
                msg = "Error while executing check function: %s" % err_str
                error_handler.collect_error(
                    "check_error", errors.SchemaError(
                        self, check_obj, msg,
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index
                    ),
                    original_exc=err
                )

        if lazy and error_handler.collected_errors:
            raise errors.SchemaErrors(
                error_handler.collected_errors, check_obj)

        assert all(check_results)
        return check_obj

    def __call__(
            self,
            check_obj: Union[pd.DataFrame, pd.Series],
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Alias for ``validate`` method."""
        return self.validate(check_obj, head, tail, sample, random_state, lazy)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SeriesSchema(SeriesSchemaBase):
    """Series validator."""

    def __init__(
            self,
            pandas_dtype: PandasDtypeInputTypes = None,
            checks: CheckList = None,
            index=None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            name: str = None) -> None:
        """Initialize series schema base object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a
            pandas.Series object.
        :type checks: callable
        :param index: specify the datatypes and properties of the index.
        :param nullable: Whether or not column can contain null values.
        :type nullable: bool
        :param allow_duplicates:
        :type allow_duplicates: bool
        """
        super().__init__(
            pandas_dtype, checks, nullable, allow_duplicates, coerce, name
        )
        self.index = index

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def validate(
            self,
            check_obj: pd.Series,
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> pd.Series:
        # pylint: disable=duplicate-code
        """Validate a Series object.

        :param check_obj: One-dimensional ndarray with axis labels
            (including time series).
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrorReport``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :returns: validated Series.

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> series_schema = pa.SeriesSchema(
        ...     pa.Float, [
        ...         pa.Check(lambda s: s > 0),
        ...         pa.Check(lambda s: s < 1000),
        ...         pa.Check(lambda s: s.mean() > 300),
        ...     ])
        >>> series = pd.Series([1, 100, 800, 900, 999], dtype=float)
        >>> print(series_schema.validate(series))
        0      1.0
        1    100.0
        2    800.0
        3    900.0
        4    999.0
        dtype: float64

        """
        if not isinstance(check_obj, pd.Series):
            raise TypeError(
                "expected %s, got %s" % (pd.Series, type(check_obj)))

        if self.coerce:
            check_obj = self.coerce_dtype(check_obj)

        if self.index is not None and (self.index.coerce or self.coerce):
            check_obj.index = self.index.coerce_dtype(check_obj.index)

        # validate index
        if self.index:
            self.index(check_obj)

        return super(SeriesSchema, self).validate(
            check_obj, head, tail, sample, random_state, lazy
        )

    def __call__(
            self,
            check_obj: pd.Series,
            head: Optional[int] = None,
            tail: Optional[int] = None,
            sample: Optional[int] = None,
            random_state: Optional[int] = None,
            lazy: bool = False,
    ) -> pd.Series:
        """Alias for :func:`SeriesSchema.validate` method."""
        return self.validate(check_obj, head, tail, sample, random_state, lazy)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _pandas_obj_to_validate(
        dataframe_or_series: Union[pd.DataFrame, pd.Series],
        head: Optional[int],
        tail: Optional[int],
        sample: Optional[int],
        random_state: Optional[int]) -> Union[pd.DataFrame, pd.Series]:
    pandas_obj_subsample = []
    if head is not None:
        pandas_obj_subsample.append(dataframe_or_series.head(head))
    if tail is not None:
        pandas_obj_subsample.append(dataframe_or_series.tail(tail))
    if sample is not None:
        pandas_obj_subsample.append(
            dataframe_or_series.sample(sample, random_state=random_state))
    return dataframe_or_series if not pandas_obj_subsample else \
        pd.concat(pandas_obj_subsample).drop_duplicates()


def _handle_check_results(
        schema: Union[DataFrameSchema, SeriesSchemaBase],
        check_index: int,
        check: Union[Check, Hypothesis],
        check_obj: Union[pd.DataFrame, pd.Series],
        *check_args) -> bool:
    """Handle check results, raising SchemaError on check failure.

    :param check_index: index of check in the schema component check list.
    :param check: Check object used to validate pandas object.
    :param check_args: arguments to pass into check object.
    :returns: True if check results pass or check.raise_warning=True, otherwise
        False.
    """
    check_result = check(check_obj, *check_args)
    if not check_result.check_passed:
        if check_result.failure_cases is None:
            # encode scalar False values explicitly
            failure_cases = scalar_failure_case(check_result.check_passed)
            error_msg = format_generic_error_message(
                schema, check, check_index)
        else:
            failure_cases = reshape_failure_cases(
                check_result.failure_cases, check.ignore_na)
            error_msg = format_vectorized_error_message(
                schema, check, check_index, failure_cases)

        # raise a warning without exiting if the check is specified to do so
        if check.raise_warning:
            warnings.warn(error_msg, UserWarning)
            return True
        raise errors.SchemaError(
            schema, check_obj, error_msg,
            failure_cases=failure_cases,
            check=check,
            check_index=check_index
        )
    return check_result.check_passed
