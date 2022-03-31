"""Core pandera schema class definitions."""
# pylint: disable=too-many-lines

from __future__ import annotations

import copy
import itertools
import os
import traceback
import warnings
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd

from . import check_utils, errors
from . import strategies as st
from .checks import Check
from .dtypes import DataType
from .engines import pandas_engine
from .error_formatters import (
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
    scalar_failure_case,
)
from .error_handlers import SchemaErrorHandler
from .hypotheses import Hypothesis

if TYPE_CHECKING:
    from pandera.schema_components import Column


N_INDENT_SPACES = 4

CheckList = Optional[
    Union[Union[Check, Hypothesis], List[Union[Check, Hypothesis]]]
]

CheckListProperty = List[Union[Check, Hypothesis]]

PandasDtypeInputTypes = Union[
    str,
    type,
    DataType,
    Type,
    pd.core.dtypes.base.ExtensionDtype,
    np.dtype,
    None,
]

TSeriesSchemaBase = TypeVar("TSeriesSchemaBase", bound="SeriesSchemaBase")


def _inferred_schema_guard(method):
    """
    Invoking a method wrapped with this decorator will set _is_inferred to
    False.
    """

    @wraps(method)
    def _wrapper(schema, *args, **kwargs):
        new_schema = method(schema, *args, **kwargs)
        if new_schema is not None and id(new_schema) != id(schema):
            # if method returns a copy of the schema object,
            # the original schema instance and the copy should be set to
            # not inferred.
            new_schema._is_inferred = False
        return new_schema

    return _wrapper


class DataFrameSchema:  # pylint: disable=too-many-public-methods
    """A light-weight pandas DataFrame validator."""

    def __init__(
        self,
        columns: Optional[Dict[Any, Column]] = None,
        checks: CheckList = None,
        index=None,
        dtype: PandasDtypeInputTypes = None,
        coerce: bool = False,
        strict: Union[bool, str] = False,
        name: Optional[str] = None,
        ordered: bool = False,
        unique: Optional[Union[str, List[str]]] = None,
        unique_column_names: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize DataFrameSchema validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :type columns: mapping of column names and column schema component.
        :param checks: dataframe-wide checks.
        :param index: specify the datatypes and properties of the index.
        :param dtype: datatype of the dataframe. This overrides the data
            types specified in any of the columns. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes.
        :param coerce: whether or not to coerce all of the columns on
            validation. This has no effect on columns where
            ``dtype=None``
        :param strict: ensure that all and only the columns defined in the
            schema are present in the dataframe. If set to 'filter',
            only the columns in the schema will be passed to the validated
            dataframe. If set to filter and columns defined in the schema
            are not present in the dataframe, will throw an error.
        :param name: name of the schema.
        :param ordered: whether or not to validate the columns order.
        :param unique: a list of columns that should be jointly unique.
        :param unique_column_names: whether or not column names must be unique.
        :param title: A human-readable label for the schema.
        :param description: An arbitrary textual description of the schema.

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
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        self.columns: Dict[Any, Column] = {} if columns is None else columns

        if strict not in (
            False,
            True,
            "filter",
        ):
            raise errors.SchemaInitError(
                "strict parameter must equal either `True`, `False`, "
                "or `'filter'`."
            )

        self.checks: CheckListProperty = checks
        self.index = index
        self.strict: Union[bool, str] = strict
        self.name: Optional[str] = name
        self.dtype: PandasDtypeInputTypes = dtype  # type: ignore
        self._coerce = coerce
        self._ordered = ordered
        self._unique = unique
        self._unique_column_names = unique_column_names
        self._title = title
        self._description = description
        self._validate_schema()
        self._set_column_names()

        # this attribute is not meant to be accessed by users and is explicitly
        # set to True in the case that a schema is created by infer_schema.
        self._IS_INFERRED = False

        # This restriction can be removed once logical types are introduced:
        # https://github.com/pandera-dev/pandera/issues/788
        if not coerce and isinstance(self.dtype, pandas_engine.PydanticModel):
            raise errors.SchemaInitError(
                "Specifying a PydanticModel type requires coerce=True."
            )

    @property
    def coerce(self) -> bool:
        """Whether to coerce series to specified type."""
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

    @property
    def ordered(self):
        """Whether or not to validate the columns order."""
        return self._ordered

    @ordered.setter
    def ordered(self, value: bool) -> None:
        """Set ordered attribute"""
        self._ordered = value

    @property
    def unique_column_names(self):
        """Whether multiple columns with the same name can be present."""
        return self._unique_column_names

    @unique_column_names.setter
    def unique_column_names(self, value: bool) -> None:
        """Set allow_duplicated_column_names attribute"""
        self._unique_column_names = value

    @property
    def title(self):
        """A human-readable label for the schema."""
        return self._title

    @property
    def description(self):
        """An arbitrary textual description of the schema."""
        return self._description

    # the _is_inferred getter and setter methods are not public
    @property
    def _is_inferred(self) -> bool:
        return self._IS_INFERRED

    @_is_inferred.setter
    def _is_inferred(self, value: bool) -> None:
        self._IS_INFERRED = value

    def _validate_schema(self) -> None:
        for column_name, column in self.columns.items():
            for check in column.checks:
                if check.groupby is None or callable(check.groupby):
                    continue
                nonexistent_groupby_columns = [
                    c for c in check.groupby if c not in self.columns
                ]
                if nonexistent_groupby_columns:
                    raise errors.SchemaInitError(
                        f"groupby argument {nonexistent_groupby_columns} in "
                        f"Check for Column {column_name} not "
                        "specified in the DataFrameSchema."
                    )

    def _set_column_names(self) -> None:
        def _set_column_handler(column, column_name):
            if column.name is not None and column.name != column_name:
                warnings.warn(
                    f"resetting column for {column} to '{column_name}'."
                )
            elif column.name == column_name:
                return column
            return column.set_name(column_name)

        self.columns = {
            column_name: _set_column_handler(column, column_name)
            for column_name, column in self.columns.items()
        }

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

    def get_dtypes(self, dataframe: pd.DataFrame) -> Dict[str, DataType]:
        """
        Same as the ``dtype`` property, but expands columns where
        ``regex == True`` based on the supplied dataframe.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_dtype = {}
        for _, column in self.columns.items():
            if column.regex:
                regex_dtype.update(
                    {
                        c: column.dtype
                        for c in column.get_regex_columns(dataframe.columns)
                    }
                )
        return {
            **{n: c.dtype for n, c in self.columns.items() if not c.regex},
            **regex_dtype,
        }

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: PandasDtypeInputTypes) -> None:
        """Set the pandas dtype property."""
        self._dtype = pandas_engine.Engine.dtype(value) if value else None

    def _coerce_dtype(self, obj: pd.DataFrame) -> pd.DataFrame:
        if self.dtype is None:
            raise ValueError(
                "dtype argument is None. Must specify this argument "
                "to coerce dtype"
            )

        try:
            return self.dtype.try_coerce(obj)
        except errors.ParserError as exc:
            raise errors.SchemaError(
                self,
                obj,
                (
                    f"Error while coercing '{self.name}' to type "
                    f"{self.dtype}: {exc}\n{exc.failure_cases}"
                ),
                failure_cases=exc.failure_cases,
                check=f"coerce_dtype('{self.dtype}')",
            ) from exc

    def coerce_dtype(self, obj: pd.DataFrame) -> pd.DataFrame:
        """Coerce dataframe to the type specified in dtype.

        :param obj: dataframe to coerce.
        :returns: dataframe with coerced dtypes
        """
        error_handler = SchemaErrorHandler(lazy=True)

        def _try_coercion(coerce_fn, obj):
            try:
                return coerce_fn(obj)
            except errors.SchemaError as exc:
                error_handler.collect_error("dtype_coercion_error", exc)
                return obj

        for colname, col_schema in self.columns.items():
            if col_schema.regex:
                try:
                    matched_columns = col_schema.get_regex_columns(obj.columns)
                except errors.SchemaError:
                    matched_columns = pd.Index([])

                for matched_colname in matched_columns:
                    if col_schema.coerce or self.coerce:
                        obj[matched_colname] = _try_coercion(
                            col_schema.coerce_dtype, obj[matched_colname]
                        )
            elif (
                (col_schema.coerce or self.coerce)
                and self.dtype is None
                and colname in obj
            ):
                obj[colname] = _try_coercion(
                    col_schema.coerce_dtype, obj[colname]
                )

        if self.dtype is not None:
            obj = _try_coercion(self._coerce_dtype, obj)
        if self.index is not None and (self.index.coerce or self.coerce):
            index_schema = copy.deepcopy(self.index)
            if self.coerce:
                # coercing at the dataframe-level should apply index coercion
                # for both single- and multi-indexes.
                index_schema._coerce = True
            coerced_index = _try_coercion(index_schema.coerce_dtype, obj.index)
            if coerced_index is not None:
                obj.index = coerced_index

        if error_handler.collected_errors:
            raise errors.SchemaErrors(error_handler.collected_errors, obj)

        return obj

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
        """Check if all columns in a dataframe have a column in the Schema.

        :param pd.DataFrame check_obj: the dataframe to be validated.
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

        if not check_utils.is_table(check_obj):
            raise TypeError(f"expected pd.DataFrame, got {type(check_obj)}")

        if hasattr(check_obj, "dask"):
            # special case for dask dataframes
            if inplace:
                check_obj = check_obj.pandera.add_schema(self)
            else:
                check_obj = check_obj.copy()

            check_obj = check_obj.map_partitions(
                self._validate,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                meta=check_obj,
            )

            return check_obj.pandera.add_schema(self)

        return self._validate(
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    def _validate(
        self,
        check_obj: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements

        if self._is_inferred:
            warnings.warn(
                f"This {type(self)} is an inferred schema that hasn't been "
                "modified. It's recommended that you refine the schema "
                "by calling `add_columns`, `remove_columns`, or "
                "`update_columns` before using it to validate data.",
                UserWarning,
            )

        error_handler = SchemaErrorHandler(lazy)

        if not inplace:
            check_obj = check_obj.copy()

        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(self)

        # dataframe strictness check makes sure all columns in the dataframe
        # are specified in the dataframe schema
        if self.strict or self.ordered:
            column_names: List[Any] = []
            for col_name, col_schema in self.columns.items():
                if col_schema.regex:
                    try:
                        column_names.extend(
                            col_schema.get_regex_columns(check_obj.columns)
                        )
                    except errors.SchemaError:
                        pass
                elif col_name in check_obj.columns:
                    column_names.append(col_name)
            # ordered "set" of columns
            sorted_column_names = iter(dict.fromkeys(column_names))
            expanded_column_names = frozenset(column_names)

            # drop adjacent duplicated column names
            if check_obj.columns.has_duplicates:
                columns = [k for k, _ in itertools.groupby(check_obj.columns)]
            else:
                columns = check_obj.columns

            filter_out_columns = []

            for column in columns:
                is_schema_col = column in expanded_column_names
                if (self.strict is True) and not is_schema_col:
                    msg = (
                        f"column '{column}' not in DataFrameSchema"
                        f" {self.columns}"
                    )
                    error_handler.collect_error(
                        "column_not_in_schema",
                        errors.SchemaError(
                            self,
                            check_obj,
                            msg,
                            failure_cases=scalar_failure_case(column),
                            check="column_in_schema",
                        ),
                    )
                if self.strict == "filter" and not is_schema_col:
                    filter_out_columns.append(column)
                if self.ordered and is_schema_col:
                    try:
                        next_ordered_col = next(sorted_column_names)
                    except StopIteration:
                        pass
                    if next_ordered_col != column:
                        error_handler.collect_error(
                            "column_not_ordered",
                            errors.SchemaError(
                                self,
                                check_obj,
                                message=f"column '{column}' out-of-order",
                                failure_cases=scalar_failure_case(column),
                                check="column_ordered",
                            ),
                        )

        if self.strict == "filter":
            check_obj.drop(labels=filter_out_columns, inplace=True, axis=1)

        if self._unique_column_names:
            failed = check_obj.columns[check_obj.columns.duplicated()]
            if failed.any():
                msg = (
                    "dataframe contains multiple columns with label(s): "
                    f"{failed.tolist()}"
                )
                error_handler.collect_error(
                    "duplicate_dataframe_column_labels",
                    errors.SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=scalar_failure_case(failed),
                        check="dataframe_column_labels_unique",
                    ),
                )

        # check for columns that are not in the dataframe and collect columns
        # that are not in the dataframe that should be excluded for lazy
        # validation
        lazy_exclude_columns = []
        for colname, col_schema in self.columns.items():
            if (
                not col_schema.regex
                and colname not in check_obj
                and col_schema.required
            ):
                if lazy:
                    lazy_exclude_columns.append(colname)
                msg = (
                    f"column '{colname}' not in dataframe\n{check_obj.head()}"
                )
                error_handler.collect_error(
                    "column_not_in_dataframe",
                    errors.SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=scalar_failure_case(colname),
                        check="column_in_dataframe",
                    ),
                )

        # coerce data types
        if (
            self.coerce
            or (self.index is not None and self.index.coerce)
            or any(col.coerce for col in self.columns.values())
        ):
            try:
                check_obj = self.coerce_dtype(check_obj)
            except errors.SchemaErrors as err:
                for schema_error_dict in err.schema_errors:
                    if not lazy:
                        # raise the first error immediately if not doing lazy
                        # validation
                        raise schema_error_dict["error"]
                    error_handler.collect_error(
                        "schema_component_check", schema_error_dict["error"]
                    )

        # collect schema components for validation
        schema_components = []
        for col_name, col in self.columns.items():
            if (
                col.required or col_name in check_obj
            ) and col_name not in lazy_exclude_columns:
                if self.dtype is not None:
                    # override column dtype with dataframe dtype
                    col = copy.deepcopy(col)
                    col.dtype = self.dtype
                schema_components.append(col)

        if self.index is not None:
            schema_components.append(self.index)

        df_to_validate = _pandas_obj_to_validate(
            check_obj, head, tail, sample, random_state
        )

        check_results = []
        # schema-component-level checks
        for schema_component in schema_components:
            try:
                result = schema_component(
                    df_to_validate,
                    lazy=lazy,
                    # don't make a copy of the data
                    inplace=True,
                )
                check_results.append(check_utils.is_table(result))
            except errors.SchemaError as err:
                error_handler.collect_error("schema_component_check", err)
            except errors.SchemaErrors as err:
                for schema_error_dict in err.schema_errors:
                    error_handler.collect_error(
                        "schema_component_check", schema_error_dict["error"]
                    )

        # dataframe-level checks
        for check_index, check in enumerate(self.checks):
            try:
                check_results.append(
                    _handle_check_results(
                        self, check_index, check, df_to_validate
                    )
                )
            except errors.SchemaError as err:
                error_handler.collect_error("dataframe_check", err)

        if self.unique:
            # NOTE: fix this pylint error
            # pylint: disable=not-an-iterable
            temp_unique: List[List] = (
                [self.unique]
                if all(isinstance(x, str) for x in self.unique)
                else self.unique
            )
            for lst in temp_unique:
                duplicates = df_to_validate.duplicated(subset=lst, keep=False)
                if duplicates.any():
                    # NOTE: this is a hack to support pyspark.pandas, need to
                    # figure out a workaround to error: "Cannot combine the
                    # series or dataframe because it comes from a different
                    # dataframe."
                    if type(duplicates).__module__.startswith(
                        "pyspark.pandas"
                    ):
                        # pylint: disable=import-outside-toplevel
                        import pyspark.pandas as ps

                        with ps.option_context(
                            "compute.ops_on_diff_frames", True
                        ):
                            failure_cases = df_to_validate.loc[duplicates, lst]
                    else:
                        failure_cases = df_to_validate.loc[duplicates, lst]

                    failure_cases = reshape_failure_cases(failure_cases)
                    error_handler.collect_error(
                        "duplicates",
                        errors.SchemaError(
                            self,
                            check_obj,
                            f"columns '{*lst,}' not unique:\n{failure_cases}",
                            failure_cases=failure_cases,
                            check="multiple_fields_uniqueness",
                        ),
                    )

        if lazy and error_handler.collected_errors:
            raise errors.SchemaErrors(
                error_handler.collected_errors, check_obj
            )

        assert all(check_results), "all check results must be True."
        return check_obj

    def __call__(
        self,
        dataframe: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
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
            f"index={self.index.__repr__()}, "
            f"coerce={self.coerce}, "
            f"dtype={self._dtype}, "
            f"strict={self.strict}, "
            f"name={self.name}, "
            f"ordered={self.ordered}, "
            f"unique_column_names={self.unique_column_names}"
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
            f"{indent}coerce={self.coerce},\n"
            f"{indent}dtype={self._dtype},\n"
            f"{indent}index={index},\n"
            f"{indent}strict={self.strict}\n"
            f"{indent}name={self.name},\n"
            f"{indent}ordered={self.ordered},\n"
            f"{indent}unique_column_names={self.unique_column_names}\n"
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
    ) -> pd.DataFrame:
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

    @_inferred_schema_guard
    def add_columns(
        self, extra_schema_cols: Dict[str, Any]
    ) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
        )>

        .. seealso:: :func:`remove_columns`

        """
        schema_copy = copy.deepcopy(self)
        schema_copy.columns = {
            **schema_copy.columns,
            **DataFrameSchema(extra_schema_cols).columns,
        }
        return schema_copy

    @_inferred_schema_guard
    def remove_columns(self, cols_to_remove: List[str]) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
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

        return schema_copy

    @_inferred_schema_guard
    def update_column(self, column_name: str, **kwargs) -> "DataFrameSchema":
        """Create copy of a :class:`DataFrameSchema` with updated column
        properties.

        :param column_name:
        :param kwargs: key-word arguments supplied to
            :class:`~pandera.schema_components.Column`
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
        )>

        .. seealso:: :func:`rename_columns`

        """
        # check that columns exist in schema

        if "name" in kwargs:
            raise ValueError("cannot update 'name' of the column.")
        if column_name not in self.columns:
            raise ValueError(f"column '{column_name}' not in {self}")
        schema_copy = copy.deepcopy(self)
        column_copy = copy.deepcopy(self.columns[column_name])
        new_column = column_copy.__class__(
            **{**column_copy.properties, **kwargs}
        )
        schema_copy.columns.update({column_name: new_column})
        return schema_copy

    def update_columns(
        self, update_dict: Dict[str, Dict[str, Any]]
    ) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
        )>

        """

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

        return new_schema

    def rename_columns(self, rename_dict: Dict[str, str]) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
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

        return new_schema

    def select_columns(self, columns: List[Any]) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
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
        return new_schema

    def to_script(self, fp: Union[str, Path] = None) -> "DataFrameSchema":
        """Create DataFrameSchema from yaml file.

        :param path: str, Path to write script
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        import pandera.io

        return pandera.io.to_script(self, fp)

    @classmethod
    def from_yaml(cls, yaml_schema) -> "DataFrameSchema":
        """Create DataFrameSchema from yaml file.

        :param yaml_schema: str, Path to yaml schema, or serialized yaml
            string.
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        import pandera.io

        return pandera.io.from_yaml(yaml_schema)

    @overload
    def to_yaml(self, stream: None = None) -> str:  # pragma: no cover
        ...

    @overload
    def to_yaml(self, stream: os.PathLike) -> None:  # pragma: no cover
        ...

    def to_yaml(self, stream: Optional[os.PathLike] = None) -> Optional[str]:
        """Write DataFrameSchema to yaml file.

        :param stream: file stream to write to. If None, dumps to string.
        :returns: yaml string if stream is None, otherwise returns None.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        import pandera.io

        return pandera.io.to_yaml(self, stream=stream)

    def set_index(
        self, keys: List[str], drop: bool = True, append: bool = False
    ) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=<Schema Index(name=category, type=DataType(str))>,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
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
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
        )>

        .. seealso:: :func:`reset_index`

        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from pandera.schema_components import Index, MultiIndex

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
            else list(new_schema.index.indexes)
            if isinstance(new_schema.index, MultiIndex) and append
            else [new_schema.index]
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

        return new_schema

    def reset_index(
        self, level: List[str] = None, drop: bool = False
    ) -> "DataFrameSchema":
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
            coerce=False,
            dtype=None,
            index=None,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
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
            coerce=False,
            dtype=None,
            index=<Schema Index(name=unique_id2, type=DataType(str))>,
            strict=False
            name=None,
            ordered=False,
            unique_column_names=False
        )>

        .. seealso:: :func:`set_index`

        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from pandera.schema_components import Column, Index, MultiIndex

        new_schema = copy.deepcopy(self)

        if new_schema.index is None:
            raise errors.SchemaInitError(
                "There is currently no index set for this schema."
            )

        # ensure no duplicates
        level_temp: Union[List[Any], List[str]] = (
            list(set(level)) if level is not None else []
        )

        # ensure all specified keys are present in the index
        level_not_in_index: Union[List[Any], List[str], None] = (
            [x for x in level_temp if x not in new_schema.index.names]
            if isinstance(new_schema.index, MultiIndex) and level_temp
            else []
            if isinstance(new_schema.index, Index)
            and (level_temp == [new_schema.index.name])
            else level_temp
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
            else Index(
                dtype=new_index.columns[list(new_index.columns)[0]].dtype,
                checks=new_index.columns[list(new_index.columns)[0]].checks,
                nullable=new_index.columns[
                    list(new_index.columns)[0]
                ].nullable,
                unique=new_index.columns[list(new_index.columns)[0]].unique,
                coerce=new_index.columns[list(new_index.columns)[0]].coerce,
                name=new_index.columns[list(new_index.columns)[0]].name,
            )
            if (len(list(new_index.columns)) == 1) and (new_index is not None)
            else None
            if (len(list(new_index.columns)) == 0) and (new_index is not None)
            else new_index
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

    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validate

    @classmethod
    def _pydantic_validate(cls, schema: Any) -> "DataFrameSchema":
        """Verify that the input is a compatible DataFrameSchema."""
        if not isinstance(schema, cls):  # type: ignore
            raise TypeError(f"{schema} is not a {cls}.")

        return cast("DataFrameSchema", schema)


class SeriesSchemaBase:
    """Base series validator object."""

    def __init__(
        self,
        dtype: PandasDtypeInputTypes = None,
        checks: CheckList = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        name: Any = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize series schema base object.

        :param dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a
            pandas.Series object.
        :param nullable: Whether or not column can contain null values.
        :param unique: Whether or not column can contain duplicate
            values.
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param name: column name in dataframe to validate.
        :param title: A human-readable label for the series.
        :param description: An arbitrary textual description of the series.
        :type nullable: bool
        """
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]

        self.dtype = dtype  # type: ignore
        self._nullable = nullable
        self._coerce = coerce
        self._checks = checks
        self._name = name
        self._unique = unique
        self._title = title
        self._description = description

        for check in self.checks:
            if check.groupby is not None and not self._allow_groupby:
                raise errors.SchemaInitError(
                    f"Cannot use groupby checks with type {type(self)}"
                )

        # make sure pandas dtype is valid
        self.dtype  # pylint: disable=pointless-statement

        # this attribute is not meant to be accessed by users and is explicitly
        # set to True in the case that a schema is created by infer_schema.
        self._IS_INFERRED = False

        if isinstance(self.dtype, pandas_engine.PydanticModel):
            raise errors.SchemaInitError(
                "PydanticModel dtype can only be specified as a "
                "DataFrameSchema dtype."
            )

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
    def unique(self) -> bool:
        """Whether to check for duplicates in check object"""
        return self._unique

    @unique.setter
    def unique(self, value: bool) -> None:
        """Set unique attribute"""
        self._unique = value

    @property
    def coerce(self) -> bool:
        """Whether to coerce series to specified type."""
        return self._coerce

    @coerce.setter
    def coerce(self, value: bool) -> None:
        """Set coerce attribute."""
        self._coerce = value

    @property
    def name(self) -> Union[str, None]:
        """Get SeriesSchema name."""
        return self._name

    @property
    def title(self):
        """A human-readable label for the series."""
        return self._title

    @property
    def description(self):
        """An arbitrary textual description of the series."""
        return self._description

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the pandas dtype"""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: PandasDtypeInputTypes) -> None:
        """Set the pandas dtype"""
        self._dtype = pandas_engine.Engine.dtype(value) if value else None

    def coerce_dtype(self, obj: Union[pd.Series, pd.Index]) -> pd.Series:
        """Coerce type of a pd.Series by type specified in dtype.

        :param pd.Series series: One-dimensional ndarray with axis labels
            (including time series).
        :returns: ``Series`` with coerced data type
        """
        if self.dtype is None:
            return obj

        try:
            return self.dtype.try_coerce(obj)
        except errors.ParserError as exc:
            msg = (
                f"Error while coercing '{self.name}' to type "
                f"{self.dtype}: {exc}:\n{exc.failure_cases}"
            )
            raise errors.SchemaError(
                self,
                obj,
                msg,
                failure_cases=exc.failure_cases,
                check=f"coerce_dtype('{self.dtype}')",
            ) from exc

    @property
    def _allow_groupby(self):
        """Whether the schema or schema component allows groupby operations."""
        raise NotImplementedError(  # pragma: no cover
            "The _allow_groupby property must be implemented by subclasses "
            "of SeriesSchemaBase"
        )

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
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated DataFrame or Series.

        """

        if self._is_inferred:
            warnings.warn(
                f"This {type(self)} is an inferred schema that hasn't been "
                "modified. It's recommended that you refine the schema "
                "by calling `set_checks` before using it to validate data.",
                UserWarning,
            )

        error_handler = SchemaErrorHandler(lazy)

        if not inplace:
            check_obj = check_obj.copy()

        series = (
            check_obj
            if check_utils.is_field(check_obj)
            else check_obj[self.name]
        )

        series = _pandas_obj_to_validate(
            series, head, tail, sample, random_state
        )

        check_obj = _pandas_obj_to_validate(
            check_obj, head, tail, sample, random_state
        )

        if self.name is not None and series.name != self._name:
            msg = (
                f"Expected {type(self)} to have name '{self._name}', found "
                f"'{series.name}'"
            )
            error_handler.collect_error(
                "wrong_field_name",
                errors.SchemaError(
                    self,
                    check_obj,
                    msg,
                    failure_cases=scalar_failure_case(series.name),
                    check=f"field_name('{self._name}')",
                ),
            )

        if not self._nullable:
            nulls = series.isna()
            if nulls.sum() > 0:
                failed = series[nulls]
                msg = (
                    f"non-nullable series '{series.name}' contains null "
                    f"values:\n{failed}"
                )
                error_handler.collect_error(
                    "series_contains_nulls",
                    errors.SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=reshape_failure_cases(
                            series[nulls], ignore_na=False
                        ),
                        check="not_nullable",
                    ),
                )

        # Check if the series contains duplicate values
        if self._unique:
            if type(series).__module__.startswith("pyspark.pandas"):
                duplicates = (
                    series.to_frame().duplicated().reindex(series.index)
                )
                # pylint: disable=import-outside-toplevel
                import pyspark.pandas as ps

                with ps.option_context("compute.ops_on_diff_frames", True):
                    failed = series[duplicates]
            else:
                duplicates = series.duplicated()
                failed = series[duplicates]

            if duplicates.any():
                msg = (
                    f"series '{series.name}' contains duplicate values:\n"
                    f"{failed}"
                )
                error_handler.collect_error(
                    "series_contains_duplicates",
                    errors.SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=reshape_failure_cases(failed),
                        check="field_uniqueness",
                    ),
                )

        if self._dtype is not None and (
            not self._dtype.check(pandas_engine.Engine.dtype(series.dtype))
        ):
            msg = (
                f"expected series '{series.name}' to have type {self._dtype}, "
                + f"got {series.dtype}"
            )
            error_handler.collect_error(
                "wrong_dtype",
                errors.SchemaError(
                    self,
                    check_obj,
                    msg,
                    failure_cases=scalar_failure_case(str(series.dtype)),
                    check=f"dtype('{self.dtype}')",
                ),
            )

        check_results = []
        if check_utils.is_field(check_obj):
            check_obj, check_args = series, [None]
        else:
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
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the
                # Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                msg = (
                    f"Error while executing check function: {err_str}\n"
                    + traceback.format_exc()
                )
                error_handler.collect_error(
                    "check_error",
                    errors.SchemaError(
                        self,
                        check_obj,
                        msg,
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )

        if lazy and error_handler.collected_errors:
            raise errors.SchemaErrors(
                error_handler.collected_errors, check_obj
            )

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
        inplace: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Alias for ``validate`` method."""
        return self.validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @st.strategy_import_error
    def strategy(self, *, size=None):
        """Create a ``hypothesis`` strategy for generating a Series.

        :param size: number of elements to generate
        :returns: a strategy that generates pandas Series objects.
        """
        return st.series_strategy(
            self.dtype,
            checks=self.checks,
            nullable=self.nullable,
            unique=self.unique,
            name=self.name,
            size=size,
        )

    def example(self, size=None) -> pd.Series:
        """Generate an example of a particular size.

        :param size: number of elements in the generated Series.
        :returns: pandas Series object.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(size=size).example()

    def __repr__(self):
        return (
            f"<Schema {self.__class__.__name__}"
            f"(name={self._name}, type={self.dtype!r})>"
        )

    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validate

    @classmethod
    def _pydantic_validate(  # type: ignore
        cls: TSeriesSchemaBase, schema: Any
    ) -> TSeriesSchemaBase:
        """Verify that the input is a compatible DataFrameSchema."""
        if not isinstance(schema, cls):  # type: ignore
            raise TypeError(f"{schema} is not a {cls}.")

        return cast(TSeriesSchemaBase, schema)


class SeriesSchema(SeriesSchemaBase):
    """Series validator."""

    def __init__(
        self,
        dtype: PandasDtypeInputTypes = None,
        checks: CheckList = None,
        index=None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        name: str = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize series schema base object.

        :param dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a
            pandas.Series object.
        :param index: specify the datatypes and properties of the index.
        :param nullable: Whether or not column can contain null values.
        :param unique: Whether or not column can contain duplicate
            values.
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param name: series name.
        :param title: A human-readable label for the series.
        :param description: An arbitrary textual description of the series.

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
        inplace: bool = False,
    ) -> pd.Series:
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
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated Series.

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> series_schema = pa.SeriesSchema(
        ...     float, [
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
        if not check_utils.is_field(check_obj):
            raise TypeError(f"expected pd.Series, got {type(check_obj)}")

        if hasattr(check_obj, "dask"):
            # special case for dask series
            if inplace:
                check_obj = check_obj.pandera.add_schema(self)
            else:
                check_obj = check_obj.copy()

            check_obj = check_obj.map_partitions(
                self._validate,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                meta=check_obj,
            )

            return check_obj.pandera.add_schema(self)

        return self._validate(
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    def _validate(
        self,
        check_obj: pd.Series,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.Series:
        if not inplace:
            check_obj = check_obj.copy()

        if hasattr(check_obj, "pandera"):
            check_obj = check_obj.pandera.add_schema(self)
        error_handler = SchemaErrorHandler(lazy=lazy)

        if self.coerce:
            try:
                check_obj = self.coerce_dtype(check_obj)
                if hasattr(check_obj, "pandera"):
                    check_obj = check_obj.pandera.add_schema(self)
            except errors.SchemaError as exc:
                error_handler.collect_error("dtype_coercion_error", exc)

        # validate index
        if self.index:
            # coerce data type using index schema copy to prevent mutation
            # of original index schema attribute.
            _index = copy.deepcopy(self.index)
            _index.coerce = _index.coerce or self.coerce
            try:
                check_obj = _index(
                    check_obj, head, tail, sample, random_state, lazy, inplace
                )
            except errors.SchemaError as exc:
                error_handler.collect_error("dtype_coercion_error", exc)
            except errors.SchemaErrors as err:
                for schema_error_dict in err.schema_errors:
                    error_handler.collect_error(
                        "index_check", schema_error_dict["error"]
                    )
        # validate series
        try:
            super().validate(
                check_obj, head, tail, sample, random_state, lazy, inplace
            )
        except errors.SchemaErrors as err:
            for schema_error_dict in err.schema_errors:
                error_handler.collect_error(
                    "series_check", schema_error_dict["error"]
                )

        if error_handler.collected_errors:
            raise errors.SchemaErrors(
                error_handler.collected_errors, check_obj
            )

        return check_obj

    def __call__(
        self,
        check_obj: pd.Series,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.Series:
        """Alias for :func:`SeriesSchema.validate` method."""
        return self.validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _pandas_obj_to_validate(
    dataframe_or_series: Union[pd.DataFrame, pd.Series],
    head: Optional[int],
    tail: Optional[int],
    sample: Optional[int],
    random_state: Optional[int],
) -> Union[pd.DataFrame, pd.Series]:
    pandas_obj_subsample = []
    if head is not None:
        pandas_obj_subsample.append(dataframe_or_series.head(head))
    if tail is not None:
        pandas_obj_subsample.append(dataframe_or_series.tail(tail))
    if sample is not None:
        pandas_obj_subsample.append(
            dataframe_or_series.sample(sample, random_state=random_state)
        )
    return (
        dataframe_or_series
        if not pandas_obj_subsample
        else pd.concat(pandas_obj_subsample).pipe(
            lambda x: x[~x.index.duplicated()]
        )
    )


def _handle_check_results(
    schema: Union[DataFrameSchema, SeriesSchemaBase],
    check_index: int,
    check: Union[Check, Hypothesis],
    check_obj: Union[pd.DataFrame, pd.Series],
    *check_args,
) -> bool:
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
                schema, check, check_index
            )
        else:
            failure_cases = reshape_failure_cases(
                check_result.failure_cases, check.ignore_na
            )
            error_msg = format_vectorized_error_message(
                schema, check, check_index, failure_cases
            )

        # raise a warning without exiting if the check is specified to do so
        if check.raise_warning:
            warnings.warn(error_msg, UserWarning)
            return True
        raise errors.SchemaError(
            schema,
            check_obj,
            error_msg,
            failure_cases=failure_cases,
            check=check,
            check_index=check_index,
            check_output=check_result.check_output,
        )
    return check_result.check_passed
