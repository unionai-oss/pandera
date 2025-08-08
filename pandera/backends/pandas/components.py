"""Backend implementation for pandas schema components."""

import traceback
from copy import deepcopy
from typing import Any, List, Optional, Set, Tuple, Union
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.pandas.types import (
    is_field,
    is_index,
    is_multiindex,
    is_table,
)
from pandera.backends.base import CoreCheckResult
from pandera.backends.pandas.array import ArraySchemaBackend
from pandera.backends.pandas.base import PandasSchemaBackend
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import (
    ValidationScope,
    validate_scope,
    validation_type,
)


class ColumnBackend(ArraySchemaBackend):
    """Backend implementation for pandas dataframe columns."""

    def validate(
        self,
        check_obj: pd.DataFrame,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Validation backend implementation for pandas dataframe columns."""
        if not inplace:
            check_obj = check_obj.copy()

        error_handler = ErrorHandler(lazy)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        if schema.name is None:
            raise SchemaError(
                schema,
                check_obj,
                "column name is set to None. Pass the ``name`` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.",
                reason_code=SchemaErrorReason.INVALID_COLUMN_NAME,
            )

        def validate_column(check_obj, column_name, return_check_obj=False):
            try:

                # make sure the schema component mutations are reverted after
                # validation
                _orig_name = schema.name
                validated_check_obj = super(ColumnBackend, self).validate(
                    check_obj,
                    schema.set_name(column_name),
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=inplace,
                )
                # revert the schema component mutations
                schema.name = _orig_name

                if return_check_obj:
                    return validated_check_obj

            except SchemaErrors as errs:
                for err in errs.schema_errors:
                    err.column_name = column_name
                    error_handler.collect_error(
                        validation_type(err.reason_code), err.reason_code, err
                    )
            except SchemaError as err:
                err.column_name = column_name
                error_handler.collect_error(
                    validation_type(err.reason_code), err.reason_code, err
                )

        column_keys_to_check = (
            self.get_regex_columns(schema, check_obj)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            if pd.notna(schema.default):
                check_obj[column_name] = check_obj[column_name].fillna(
                    schema.default
                )
            if schema.coerce:
                try:
                    check_obj[column_name] = self.coerce_dtype(
                        check_obj[column_name],
                        schema=schema,
                    )
                except SchemaErrors as exc:
                    error_handler.collect_errors(exc.schema_errors)

            if is_table(check_obj[column_name]):
                for i in range(check_obj[column_name].shape[1]):
                    validated_column = validate_column(
                        check_obj[column_name].iloc[:, [i]],
                        column_name,
                        return_check_obj=True,
                    )
                    if schema.parsers:
                        check_obj[column_name] = validated_column
            else:
                if getattr(schema, "drop_invalid_rows", False):
                    # replace the check_obj with the validated
                    check_obj = validate_column(
                        check_obj, column_name, return_check_obj=True
                    )

                validated_column = validate_column(
                    check_obj,
                    column_name,
                    return_check_obj=True,
                )
                if schema.parsers:
                    check_obj[column_name] = validated_column

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    def get_regex_columns(self, schema, check_obj) -> Iterable:
        """Get matching column names based on regex column name pattern.

        :param schema: schema specification to use
        :param columns: columns to regex pattern match
        :returns: matching columns
        """
        columns = check_obj.columns
        if isinstance(schema.name, tuple):
            # handle MultiIndex case
            if len(schema.name) != columns.nlevels:
                raise IndexError(
                    f"Column regex name='{schema.name}' is a tuple, expected a "
                    f"MultiIndex columns with {len(schema.name)} number of "
                    f"levels, found {columns.nlevels} level(s)"
                )
            matches = np.ones(len(columns)).astype(bool)
            for i, name in enumerate(schema.name):
                matched = pd.Index(
                    columns.get_level_values(i).astype(str).str.match(name)
                ).fillna(False)
                matches = matches & np.array(matched.tolist())
            column_keys_to_check = columns[matches]
        else:
            if is_multiindex(columns):
                raise IndexError(
                    f"Column regex name {schema.name} is a string, expected a "
                    "dataframe where the index is a pd.Index object, not a "
                    "pd.MultiIndex object"
                )
            column_keys_to_check = columns[
                # str.match will return nan values when the index value is
                # not a string.
                pd.Index(columns.astype(str).str.match(schema.name))
                .fillna(False)
                .tolist()
            ]
        if column_keys_to_check.shape[0] == 0:
            raise SchemaError(
                schema=schema,
                data=columns,
                message=(
                    f"Column regex name='{schema.name}' did not match any "
                    "columns in the dataframe. Update the regex pattern so "
                    f"that it matches at least one column:\n{columns.tolist()}",
                ),
                failure_cases=str(columns.tolist()),
                check=f"no_regex_column_match('{schema.name}')",
                reason_code=SchemaErrorReason.INVALID_COLUMN_NAME,
            )
        # drop duplicates to account for potential duplicated columns in the
        # dataframe.
        return column_keys_to_check.drop_duplicates()

    def coerce_dtype(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema=None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Coerce dtype of a column, handling duplicate column names."""

        # TODO: use singledispatchmethod here
        if is_field(check_obj) or is_index(check_obj):
            return super().coerce_dtype(
                check_obj,
                schema=schema,
            )
        return check_obj.apply(
            lambda x: super(ColumnBackend, self).coerce_dtype(
                x,
                schema=schema,
            ),
            axis="columns",
        )

    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(self, check_obj, schema):
        check_results: list[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            check_args = [None] if is_field(check_obj) else [schema.name]
            try:
                check_results.append(
                    self.run_check(
                        check_obj, schema, check, check_index, *check_args
                    )
                )
            except SchemaError as err:
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                        message=str(err),
                        failure_cases=err.failure_cases,
                        original_exc=err,
                    )
                )
            except Exception as err:
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if err.args else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                msg = (
                    f"Error while executing check function: {err_str}\n"
                    + traceback.format_exc()
                )
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=err_str,
                        original_exc=err,
                    )
                )
        return check_results


class IndexBackend(ArraySchemaBackend):
    """Backend implementation for pandas index."""

    def validate(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        if is_multiindex(check_obj.index):
            raise SchemaError(
                schema,
                check_obj,
                "Attempting to validate mismatch index",
                reason_code=SchemaErrorReason.MISMATCH_INDEX,
            )

        error_handler = ErrorHandler(lazy)

        if schema.coerce:
            try:
                check_obj.index = schema.coerce_dtype(check_obj.index)
            except SchemaError as exc:
                error_handler.collect_error(
                    validation_type(exc.reason_code),
                    exc.reason_code,
                    exc,
                )

        try:
            _validated_obj = super().validate(
                check_obj.index.to_series().reset_index(drop=True),
                schema,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
            assert is_field(_validated_obj)
        except SchemaError as exc:
            error_handler.collect_error(
                validation_type(exc.reason_code),
                exc.reason_code,
                exc,
            )
        except SchemaErrors as exc:
            error_handler.collect_errors(exc.schema_errors, exc)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj


class MultiIndexBackend(PandasSchemaBackend):
    """Backend implementation for pandas multiindex."""

    def coerce_dtype(  # type: ignore[override]
        self,
        check_obj: pd.MultiIndex,
        schema=None,
    ) -> pd.MultiIndex:
        """Coerce type of a pd.Series by type specified in dtype.

        :param obj: multi-index to coerce.
        :returns: ``MultiIndex`` with coerced data type
        """
        assert schema is not None, "The `schema` argument must be provided."

        if not schema.coerce:
            return check_obj

        error_handler = ErrorHandler(lazy=True)

        # construct MultiIndex with coerced data types
        coerced_multi_index = {}
        for i, index in enumerate(schema.indexes):
            if all(x is None for x in schema.names):
                index_levels = [i]
            else:
                index_levels = [
                    i
                    for i, name in enumerate(check_obj.names)
                    if name == index.name
                ]
            for index_level in index_levels:
                index_array = check_obj.get_level_values(index_level)
                if index.coerce or schema._coerce:
                    try:
                        _index = deepcopy(index)
                        _index.coerce = True
                        index_array = _index.coerce_dtype(index_array)
                    except SchemaError as err:
                        error_handler.collect_error(
                            validation_type(
                                SchemaErrorReason.DATATYPE_COERCION
                            ),
                            SchemaErrorReason.DATATYPE_COERCION,
                            err,
                        )
                coerced_multi_index[index_level] = index_array

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        multiindex_cls = pd.MultiIndex
        # NOTE: this is a hack to support pyspark.pandas
        if type(check_obj).__module__.startswith("pyspark.pandas"):

            import pyspark.pandas as ps

            multiindex_cls = ps.MultiIndex

        return multiindex_cls.from_arrays(
            [
                # v.dtype may be different than 'object'.
                # - Reuse the original index array to keep the specialized dtype:
                #   v.to_numpy()  converts the array dtype to array of 'object' dtype.
                #   Thus removing the specialized index dtype required to pass a schema's
                #   index specialized dtype : eg: pandera.typing.Index(pandas.Int64Dtype)
                # - For Pyspark only, use to_numpy(), with the effect of keeping the
                #   bug open on this execution environment: At the time of writing, pyspark
                #   v3.3.0 does not provide a working implementation of v.array
                (
                    v.to_numpy()
                    if type(v).__module__.startswith("pyspark.pandas")
                    else v.array
                )
                for _, v in sorted(
                    coerced_multi_index.items(), key=lambda x: x[0]
                )
            ],
            names=check_obj.names,
        )

    def validate(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Validate DataFrame or Series MultiIndex.

        :param check_obj: pandas DataFrame or Series to validate.
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

        # Make a copy if we're not modifying inplace
        if not inplace:
            check_obj = check_obj.copy()

        # Ensure the object has a MultiIndex
        if not is_multiindex(check_obj.index):
            # Allow an exception for a *single-level* Index when the schema also
            # describes exactly one level to maintain compatibility (e.g. pyspark.pandas
            # often materializes a single-level MultiIndex as a plain Index).
            is_pyspark_index = (
                type(check_obj).__module__.startswith("pyspark.pandas")
                and hasattr(check_obj.index, "__module__")
                and check_obj.index.__module__.startswith("pyspark.pandas")
            )

            if len(schema.indexes) == 1 and (
                is_index(check_obj.index) or is_pyspark_index
            ):
                # Validate the single-level index directly using the Index schema.
                # This works for both pandas and pyspark.pandas objects and avoids
                # constructing a pandas DataFrame with a non-pandas Index.

                schema.indexes[0].validate(
                    check_obj,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )

                return check_obj

            raise SchemaError(
                schema,
                check_obj,
                "Attempting to validate mismatch index",  # same message as IndexBackend
                reason_code=SchemaErrorReason.MISMATCH_INDEX,
            )

        error_handler = ErrorHandler(lazy)

        # Coerce dtype at the multi-index level first if required. In lazy
        # mode we collect coercion errors so that validation can proceed and
        # aggregate all issues for the user.
        if schema.coerce:
            try:
                check_obj.index = self.__coerce_index(check_obj, schema, lazy)
            except (SchemaError, SchemaErrors) as exc:
                self._collect_or_raise(error_handler, exc, schema)

        # Map schema ``indexes`` definitions to concrete level positions in the
        # multi-index so that we can validate each level individually.
        level_mapping: list[tuple[int, Any]] = self._map_schema_to_levels(
            check_obj.index, schema, error_handler
        )

        # Validate the correspondence between schema index names and the actual
        # multi-index names (order and presence checks).
        self._validate_index_names(
            check_obj.index, schema, level_mapping, error_handler
        )

        # Iterate over the expected index levels and validate each level with its
        # corresponding ``Index`` schema component.
        for level_pos, index_schema in level_mapping:
            stub_df = pd.DataFrame(
                index=check_obj.index.get_level_values(level_pos)
            )
            # We've already taken care of coercion, so we can disable it now.
            index_schema = deepcopy(index_schema)
            index_schema.coerce = False

            try:
                # Validate using the schema for this level
                index_schema.validate(
                    stub_df,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except (SchemaError, SchemaErrors) as exc:
                self._collect_or_raise(error_handler, exc, schema)

        # Raise aggregated errors in lazy mode
        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    @staticmethod
    def _nonconsecutive_duplicates(
        names: list[Optional[Any]],
    ) -> list[Optional[Any]]:
        """Check whether the names have any non-consecutive duplicates.

        If any non-consecutive duplicates are found, return the names that
        are duplicated non-consecutively.
        """
        seen: set[Optional[Any]] = set()
        last_name: Optional[Any] = None
        nonconsecutive_duplicates: set[Optional[Any]] = set()
        for name in names:
            if name == last_name:
                # Consecutive duplicate – allowed.
                continue
            if name in seen and name is not None:
                # Duplicate not consecutive – violation.
                nonconsecutive_duplicates.add(name)
            seen.add(name)
            last_name = name
        return list(nonconsecutive_duplicates)

    @staticmethod
    def _collect_or_raise(
        error_handler: Optional[ErrorHandler],
        err: Union[SchemaError, SchemaErrors],
        schema,
    ) -> None:
        """Collect errors (respecting lazy), adjusting schema context and
        failure cases appropriately.
        """

        def _update_schema_error(schema_error: SchemaError):
            """Add `column` info to tabular failure cases."""

            try:
                failure_cases = schema_error.failure_cases  # may not exist
            except AttributeError:
                return

            # Replace the schema context with the top-level MultiIndex schema so
            # that downstream error reporting groups these failures under the
            # "MultiIndex" key.
            try:
                schema_error.schema = schema
            except Exception:
                # In case the attribute is frozen / read-only, skip.
                pass

            if is_table(failure_cases):
                # Attach the originating component name so that it can be
                # displayed alongside the failure row.
                component_name = getattr(schema, "name", None)
                # if component_name is not None and "column" not in failure_cases.columns:
                schema_error.failure_cases = failure_cases.assign(
                    column=component_name
                )

        # First, update failure_cases in the incoming error(s) with the component name
        if isinstance(err, SchemaErrors):
            for se in err.schema_errors:
                _update_schema_error(se)

            if error_handler is not None and error_handler.lazy:
                error_handler.collect_errors(err.schema_errors, err)
            else:
                # Fail fast with the first individual error for consistency
                raise err.schema_errors[0] from err
        else:  # Single SchemaError
            _update_schema_error(err)

            if error_handler is not None and error_handler.lazy:
                error_handler.collect_error(
                    validation_type(err.reason_code), err.reason_code, err
                )
            else:
                raise err

    def _validate_index_names(
        self,
        mi: pd.MultiIndex,
        schema,
        level_mapping: list[tuple[int, Any]],
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        """Perform high-level validation of index names/order requirements.

        When ``error_handler`` is provided and lazy mode is enabled, all
        discovered violations are collected instead of stopping at the first
        one, allowing the caller to aggregate multiple issues for the user.
        """

        names = list(mi.names)

        # Ordered validation – check that the names are in the expected order
        # and that there are no non-consecutive duplicates.
        if schema.ordered:
            nonconsecutive_duplicates = self._nonconsecutive_duplicates(names)
            for violation in nonconsecutive_duplicates:
                self._collect_or_raise(
                    error_handler,
                    SchemaError(
                        schema=schema,
                        data=mi,
                        message=f"column '{violation}' out-of-order",
                        failure_cases=violation,
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    ),
                    schema,
                )

            mapped_names = [names[level_pos] for level_pos, _ in level_mapping]

            # Ensure that schema-specified names appear in the expected order
            expected = [idx.name for idx in schema.indexes]
            no_explicit_names = not any(n is not None for n in expected)

            for pos, expected_name in enumerate(expected):
                if pos >= len(mapped_names):
                    # We already collected the error for this when building the mapping
                    continue

                actual_name = mapped_names[pos]

                if expected_name is None:
                    # Reject only if:
                    # - schema is entirely unnamed (no named components)
                    # - AND this is a new name (not None and not a continuation)
                    if (
                        no_explicit_names  # schema entirely unnamed
                        and actual_name is not None  # new name
                        and actual_name
                        not in mapped_names[:pos]  # not a continuation
                    ):
                        self._collect_or_raise(
                            error_handler,
                            SchemaError(
                                schema=schema,
                                data=mi,
                                message=f"column '{actual_name}' out-of-order",
                                failure_cases=actual_name,
                                check="column_ordered",
                                reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                            ),
                            schema,
                        )
                else:
                    # For a named index, just check that the actual name matches the expected name.
                    if actual_name != expected_name:
                        self._collect_or_raise(
                            error_handler,
                            SchemaError(
                                schema=schema,
                                data=mi,
                                message=f"column '{expected_name}' out-of-order",
                                failure_cases=expected_name,
                                check="column_ordered",
                                reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                            ),
                            schema,
                        )
        # Unordered validation – just check that all required names are present
        else:
            required_names = {
                idx.name for idx in schema.indexes if idx.name is not None
            }
            missing = required_names.difference(set(names))
            for missing_name in missing:
                self._collect_or_raise(
                    error_handler,
                    SchemaError(
                        schema=schema,
                        data=mi,
                        message=f"column '{missing_name}' not in index",
                        failure_cases=missing_name,
                        check="column_in_index",
                        reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                    ),
                    schema,
                )

    def _map_ordered_levels(  # helper for ordered=True
        self,
        mi: pd.MultiIndex,
        schema,
        error_handler: Optional[ErrorHandler] = None,
    ) -> list[tuple[int, Any]]:
        """
        Return a list of ``(level_position, index_schema)`` mappings for an
        ordered MultiIndex schema, correctly handling duplicate names and
        unnamed (name=None) schema levels.

        Rules
        -----
        1. Named schema level -> first *unused* dataframe level with the same name
           that appears **after** the previously matched level.
        2. Unnamed schema level -> the very next *unused* dataframe level,
           regardless of its name.
        3. Duplicate schema names must map to *consecutive* dataframe levels
           with that same name.  If we encounter any different name in-between
           -> out-of-order error.
        4. If the dataframe runs out of levels before the schema list is
           exhausted -> “fewer levels than expected” error.
        """
        mapping: list[tuple[int, Any]] = []
        mi_names = list(mi.names)
        n_levels = mi.nlevels
        current_level_pos: int = 0
        last_mapped_name: Optional[str] = None

        for idx_schema in schema.indexes:
            idx_name: Optional[str] = idx_schema.name

            if idx_name is None:
                # Unnamed schema index – accept next dataframe level as-is
                mapping.append((current_level_pos, idx_schema))
                last_mapped_name = mi_names[current_level_pos]
                current_level_pos += 1
                continue

            # Skip over duplicates of the *previous* name *only* if the schema
            # is expecting a *different* name next. If the schema expects the
            # same name again (duplicate schema components), we should stay on
            # the current duplicate level so it can be mapped.
            while (
                current_level_pos < n_levels
                and last_mapped_name is not None
                and mi_names[current_level_pos] == last_mapped_name
                and idx_name != last_mapped_name
            ):
                current_level_pos += 1

            # Now walk forward until we find the index name
            while (
                current_level_pos < n_levels
                and mi_names[current_level_pos] != idx_name
            ):
                # Any *other* name before we meet `idx_name` => out-of-order
                self._collect_or_raise(
                    error_handler,
                    SchemaError(
                        schema=schema,
                        data=mi,
                        message=f"column '{idx_name}' out-of-order",
                        failure_cases=idx_name,
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    ),
                    schema,
                )
                current_level_pos += 1

            if current_level_pos >= n_levels:
                # ran off the end without finding target index level
                self._collect_or_raise(
                    error_handler,
                    SchemaError(
                        schema=schema,
                        data=mi,
                        message=f"index level with name '{idx_name}' not found",
                        failure_cases=idx_name,
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    ),
                    schema,
                )
                break

            # Found the matching level
            mapping.append((current_level_pos, idx_schema))
            last_mapped_name = idx_name
            current_level_pos += 1

        return mapping

    def _map_schema_to_levels(
        self,
        mi: pd.MultiIndex,
        schema,
        error_handler: Optional[ErrorHandler] = None,
    ):
        """Map schema index definitions to concrete level positions.

        Returns a list of tuples ``(level_position, index_schema)`` while
        aggregating any discovered mapping errors via ``_collect_or_raise``.
        """

        mapping: list[tuple[int, Any]] = []
        used_levels: set[int] = set()

        if schema.ordered:
            return self._map_ordered_levels(mi, schema, error_handler)
        else:
            # Unordered
            for idx_schema in schema.indexes:
                if idx_schema.name is not None:
                    # Get the first unused level with matching name
                    candidate_levels = [
                        i
                        for i, n in enumerate(mi.names)
                        if n == idx_schema.name and i not in used_levels
                    ]
                    if not candidate_levels:
                        self._collect_or_raise(
                            error_handler,
                            SchemaError(
                                schema=schema,
                                data=mi,
                                message=f"index level with name '{idx_schema.name}' not found",
                                failure_cases=idx_schema.name,
                                check="column_in_index",
                                reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                            ),
                            schema,
                        )
                        # Cannot map, continue to next idx_schema
                        continue
                    level_pos = candidate_levels[0]
                else:
                    # Unnamed schema index - get the first unmatched level
                    remaining = [
                        i for i in range(mi.nlevels) if i not in used_levels
                    ]
                    if not remaining:
                        self._collect_or_raise(
                            error_handler,
                            SchemaError(
                                schema=schema,
                                data=mi,
                                message="Ran out of index levels to map to unnamed schema component",
                                failure_cases=str(mi.names),
                                check="column_in_index",
                                reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                            ),
                            schema,
                        )
                        continue
                    level_pos = remaining[0]
                mapping.append((level_pos, idx_schema))
                used_levels.add(level_pos)

        return mapping

    def __coerce_index(self, check_obj, schema, lazy):
        """Coerce index"""
        try:
            return self.coerce_dtype(
                check_obj.index,
                schema=schema,  # type: ignore[arg-type]
            )
        except SchemaErrors as err:
            if lazy:
                raise
            raise err.schema_errors[0] from err
