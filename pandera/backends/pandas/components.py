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
            raise SchemaDefinitionError("When drop_invalid_rows is True, lazy must be set to True.")

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
                    error_handler.collect_error(validation_type(err.reason_code), err.reason_code, err)
            except SchemaError as err:
                err.column_name = column_name
                error_handler.collect_error(validation_type(err.reason_code), err.reason_code, err)

        column_keys_to_check = self.get_regex_columns(schema, check_obj) if schema.regex else [schema.name]

        for column_name in column_keys_to_check:
            if pd.notna(schema.default):
                check_obj[column_name] = check_obj[column_name].fillna(schema.default)
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
                    check_obj = validate_column(check_obj, column_name, return_check_obj=True)

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
                matched = pd.Index(columns.get_level_values(i).astype(str).str.match(name)).fillna(False)
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
                pd.Index(columns.astype(str).str.match(schema.name)).fillna(False).tolist()
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
                check_results.append(self.run_check(check_obj, schema, check, check_index, *check_args))
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
                msg = f"Error while executing check function: {err_str}\n" + traceback.format_exc()
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
    """Backend implementation for pandas multiindex without relying on the
    dataframe-oriented validation logic. This avoids the additional memory
    overhead of materialising a temporary dataframe that mirrors the
    MultiIndex levels."""

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
                index_levels = [i for i, name in enumerate(check_obj.names) if name == index.name]
            for index_level in index_levels:
                index_array = check_obj.get_level_values(index_level)
                if index.coerce or schema._coerce:
                    try:
                        _index = deepcopy(index)
                        _index.coerce = True
                        index_array = _index.coerce_dtype(index_array)
                    except SchemaError as err:
                        error_handler.collect_error(
                            validation_type(SchemaErrorReason.DATATYPE_COERCION),
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
                (v.to_numpy() if type(v).__module__.startswith("pyspark.pandas") else v.array)
                for _, v in sorted(coerced_multi_index.items(), key=lambda x: x[0])
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
        """Validate the MultiIndex of a pandas DataFrame/Series."""

        # Ensure we are validating against a MultiIndex
        # We need to raise immediately here because there's not much we can do
        # with a non-MultiIndex index.
        if not is_multiindex(check_obj.index):
            raise SchemaError(
                schema,
                check_obj,
                "Attempting to validate mismatch index",  # same message as IndexBackend
                reason_code=SchemaErrorReason.MISMATCH_INDEX,
            )

        # Perform a copy if requested so that the original dataframe is kept
        # intact when ``inplace`` is False.
        if not inplace:
            check_obj = check_obj.copy()

        error_handler = ErrorHandler(lazy)

        # Coerce dtype at the multi-index level first if required. In lazy
        # mode we collect coercion errors so that validation can proceed and
        # aggregate all issues for the user.
        if schema.coerce:
            try:
                check_obj.index = self.__coerce_index(check_obj, schema, lazy)
            except (SchemaError, SchemaErrors) as exc:
                self._collect_or_raise(error_handler, exc)

        # Validate the correspondence between schema index names and the actual
        # multi-index names (order and presence checks).
        self._validate_index_names(check_obj.index, schema, error_handler)

        # Map schema ``indexes`` definitions to concrete level positions in the
        # multi-index so that we can validate each level individually.
        level_mapping: List[Tuple[int, Any]] = self._map_schema_to_levels(check_obj.index, schema, error_handler)

        # Iterate over the expected index levels and validate each level with its
        # corresponding ``Index`` schema component.
        for level_pos, index_schema in level_mapping:
            stub_df = pd.DataFrame(index=check_obj.index.get_level_values(level_pos))

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
                self._collect_or_raise(error_handler, exc)

        # Raise aggregated errors in lazy mode
        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    @staticmethod
    def _nonconsecutive_duplicates(names: List[Optional[str]]) -> Optional[str]:
        """Check whether the names have any non-consecutive duplicates.

        If any non-consecutive duplicates are found, return the names that
        are duplicated non-consecutively.
        """
        seen: Set[str] = set()
        last_name: Optional[str] = None
        nonconsecutive_duplicates: Set[str] = set()
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
    ) -> None:  # noqa: D401
        """Collect SchemaError/SchemaErrors* into error_handler when lazy,
        otherwise, raise the first error.
        """

        if isinstance(err, SchemaErrors):
            # Multi-error container
            if error_handler is not None and error_handler.lazy:
                error_handler.collect_errors(err.schema_errors, err)
            else:
                # Fail fast with the first individual error for consistency
                raise err.schema_errors[0] from err
        else:  # Single SchemaError
            if error_handler is not None and error_handler.lazy:
                error_handler.collect_error(validation_type(err.reason_code), err.reason_code, err)
            else:
                raise err

    def _validate_index_names(
        self,
        mi: pd.MultiIndex,
        schema,
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        """Perform high-level validation of index names/order requirements.

        When ``error_handler`` is provided (and lazy mode is enabled), all
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
                )

            # Ensure that schema-specified names appear in the expected order
            expected = [idx.name for idx in schema.indexes]

            for pos, expected_name in enumerate(expected):
                if pos >= len(names):
                    self._collect_or_raise(
                        error_handler,
                        SchemaError(
                            schema=schema,
                            data=mi,
                            message=f"MultiIndex has fewer levels than expected at position {pos}",
                            failure_cases=str(mi.names),
                            check="column_ordered",
                            reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                        ),
                    )
                    continue

                actual_name = names[pos]

                if expected_name is None:
                    if actual_name is None:
                        continue
                    if actual_name in names[:pos]:
                        # treat as duplicate continuation even if previous level had name
                        # Note that because of the nonconsecutive duplicates check,
                        # this is only possible if actual_name matches the previous non-None name
                        continue
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
                    )
                else:
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
                        )
        # Unordered validation – just check that all required names are present
        else:
            required_names = {idx.name for idx in schema.indexes if idx.name is not None}
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
                )

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

        mapping: List[Tuple[int, Any]] = []
        used_levels: set[int] = set()

        if schema.ordered:
            # Simple positional mapping when ordered.
            if len(schema.indexes) > mi.nlevels:
                self._collect_or_raise(
                    error_handler,
                    SchemaError(
                        schema=schema,
                        data=mi,
                        message="MultiIndex has fewer levels than specified in schema",
                        failure_cases=str(mi.names),
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    ),
                )
            # Map up to the minimum of schema-defined indexes and actual levels
            end_len = min(len(schema.indexes), mi.nlevels)
            for position in range(end_len):
                idx_schema = schema.indexes[position]
                mapping.append((position, idx_schema))
                used_levels.add(position)
        else:
            # Unordered – match by name first, then fallback to unused levels.
            for idx_schema in schema.indexes:
                if idx_schema.name is not None:
                    candidate_levels = [
                        i for i, n in enumerate(mi.names) if n == idx_schema.name and i not in used_levels
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
                        )
                        # Cannot map, continue to next idx_schema
                        continue
                    level_pos = candidate_levels[0]
                else:
                    remaining = [i for i in range(mi.nlevels) if i not in used_levels]
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
                        )
                        continue
                    level_pos = remaining[0]
                mapping.append((level_pos, idx_schema))
                used_levels.add(level_pos)

        return mapping

    def __coerce_index(self, check_obj, schema, lazy):
        """Helper that wraps ``coerce_dtype`` for the full multi-index."""
        try:
            return self.coerce_dtype(
                check_obj.index,
                schema=schema,  # type: ignore[arg-type]
            )
        except SchemaErrors as err:
            if lazy:
                raise
            raise err.schema_errors[0] from err
