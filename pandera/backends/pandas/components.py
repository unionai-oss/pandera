"""Backend implementation for pandas schema components."""

# pylint: disable=too-many-locals

import traceback
from copy import deepcopy
from typing import Iterable, List, Optional, Union, Any

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
from pandera.validation_depth import validation_type


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
        # pylint: disable=too-many-branches
        """Validation backend implementation for pandas dataframe columns.."""
        if not inplace:
            check_obj = check_obj.copy()

        error_handler = ErrorHandler(lazy=lazy)

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
                # pylint: disable=super-with-arguments
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
        # pylint: disable=super-with-arguments
        # pylint: disable=fixme
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

    def run_checks(self, check_obj, schema):
        check_results: List[CoreCheckResult] = []
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
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
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
        # pylint: disable=fixme
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
            # pylint: disable=import-outside-toplevel
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
        """Validate the MultiIndex of a pandas DataFrame/Series.

        Unlike the previous implementation, this method operates directly on
        the ``MultiIndex`` object and therefore avoids constructing a
        temporary helper dataframe, which could incur a large memory
        footprint when the index is big.
        """

        # Ensure we are validating against a MultiIndex
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

        # Coerce dtype at the multi-index level first if required
        if schema.coerce:
            check_obj.index = self.__coerce_index(check_obj, schema, lazy)

        error_handler = ErrorHandler(lazy)

        # Validate the correspondence between schema index names and the actual
        # multi-index names (order and presence checks)
        try:
            self._validate_index_names(check_obj.index, schema)
        except SchemaError as err:
            if lazy:
                error_handler.collect_error(validation_type(err.reason_code), err.reason_code, err)
            else:
                raise

        # Map schema ``indexes`` definitions to concrete level positions in the
        # multi-index so that we can validate each level individually.
        level_mapping: List[tuple[int, Any]] = []
        try:
            level_mapping = self._map_schema_to_levels(check_obj.index, schema)
        except SchemaError as err:
            if lazy:
                error_handler.collect_error(validation_type(err.reason_code), err.reason_code, err)
            else:
                raise

        # Iterate over the mapping and validate each index level with its
        # corresponding ``Index`` schema component.
        for level_pos, index_schema in level_mapping:
            # Check if all checks are element-wise to enable optimization
            all_elementwise = all(getattr(check, "element_wise", False) for check in index_schema.checks)

            level_values = check_obj.index.unique(level=level_pos)
            # First, handle uniqueness check using codes if required
            # This optimization works regardless of whether there are other checks
            if getattr(index_schema, "unique", False):
                # Use codes for uniqueness check - no value materialization needed
                has_duplicates = len(level_values) < len(check_obj.index)

                if has_duplicates:
                    # Get the actual duplicate values for error reporting
                    level_values = check_obj.index.get_level_values(level_pos)
                    duplicates = level_values.duplicated(keep=False)
                    raise SchemaError(
                        schema=index_schema,
                        data=check_obj,
                        message=f"index level '{index_schema.name}' contains duplicate values",
                        failure_cases=level_values[duplicates].unique(),
                        check="field_uniqueness",
                        reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                    )

            index_schema_without_unique = deepcopy(index_schema)
            index_schema_without_unique.unique = False

            # Now handle remaining validations (dtype, nullable, custom checks)
            # Choose the most efficient value set based on the types of checks
            if all_elementwise:
                # All checks are element-wise - use unique values
                # Element-wise checks work the same on each individual value
                unique_values = check_obj.index.unique(level=level_pos)
                stub_df = pd.DataFrame(index=unique_values)
            else:
                # Non-element-wise or mixed checks - use full values
                level_values = check_obj.index.get_level_values(level_pos)
                stub_df = pd.DataFrame(index=level_values)

            try:
                # Validate using the original schema
                # If codes check passed uniqueness, the regular uniqueness check will pass quickly
                index_schema_without_unique.validate(
                    stub_df,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except SchemaErrors as exc:
                if lazy:
                    error_handler.collect_errors(exc.schema_errors, exc)
                else:
                    # In non-lazy mode we raise the first error encountered.
                    raise exc.schema_errors[0] from exc
            except SchemaError as exc:
                if lazy:
                    error_handler.collect_error(validation_type(exc.reason_code), exc.reason_code, exc)
                else:
                    raise

        # Raise aggregated errors in lazy mode
        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    @staticmethod
    def _consecutive_duplicate_violation(names: List[Optional[str]]) -> Optional[str]:
        """Return the name that violates the consecutive-duplicate rule.

        For ordered multi-index schemas, the same name must appear in
        consecutive positions. If a previously seen name appears again after a
        different name has been encountered, it is considered out-of-order and
        should trigger an error.
        """
        seen: set[str] = set()
        last_name: Optional[str] = None
        for name in names:
            if name == last_name:
                # Consecutive duplicate – allowed.
                continue
            if name in seen and name is not None:
                # Duplicate not consecutive – violation.
                return name
            seen.add(name)
            last_name = name
        return None

    def _validate_index_names(self, mi: pd.MultiIndex, schema) -> None:
        """Perform high-level validation of index names/order requirements."""
        names = list(mi.names)
        # Ordered validation
        if schema.ordered:
            violation = self._consecutive_duplicate_violation(names)
            if violation is not None:
                raise SchemaError(
                    schema=schema,
                    data=mi,
                    message=f"column '{violation}' out-of-order",
                    failure_cases=violation,
                    check="column_ordered",
                    reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                )
            # Ensure that schema-specified names appear in the expected order
            expected = [idx.name for idx in schema.indexes]
            # Compare up to the length of expected list
            for pos, expected_name in enumerate(expected):
                if pos >= len(names):
                    # Not enough levels present.
                    raise SchemaError(
                        schema=schema,
                        data=mi,
                        message=f"MultiIndex has fewer levels than expected at position {pos}",
                        failure_cases=str(mi.names),
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    )

                actual_name = names[pos]

                if expected_name is None:
                    # Unnamed level in schema can correspond to any actual name.
                    if actual_name is None:
                        continue
                    # allow duplicates of previously seen names (to support consecutive duplicates)
                    if actual_name in names[:pos]:
                        # treat as duplicate continuation even if previous level had name
                        continue
                    # otherwise, new unexpected name -> out-of-order
                    raise SchemaError(
                        schema=schema,
                        data=mi,
                        message=f"column '{actual_name}' out-of-order",
                        failure_cases=actual_name,
                        check="column_ordered",
                        reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                    )
                else:
                    # Schema expects specific name.
                    if actual_name != expected_name:
                        raise SchemaError(
                            schema=schema,
                            data=mi,
                            message=f"column '{expected_name}' out-of-order",
                            failure_cases=expected_name,
                            check="column_ordered",
                            reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                        )
        # Unordered validation – only presence matters
        else:
            required_names = {idx.name for idx in schema.indexes if idx.name is not None}
            missing = required_names.difference(set(names))
            if missing:
                missing_name = next(iter(missing))
                raise SchemaError(
                    schema=schema,
                    data=mi,
                    message=f"column '{missing_name}' not in index",
                    failure_cases=missing_name,
                    check="column_in_index",
                    reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                )

    def _map_schema_to_levels(self, mi: pd.MultiIndex, schema):
        """Map schema index definitions to concrete level positions.

        Returns a list of tuples ``(level_position, index_schema)``.
        """
        mapping: List[tuple[int, Any]] = []
        used_levels: set[int] = set()

        if schema.ordered:
            # Simple positional mapping when ordered.
            if len(schema.indexes) > mi.nlevels:
                raise SchemaError(
                    schema=schema,
                    data=mi,
                    message="MultiIndex has fewer levels than specified in schema",
                    failure_cases=str(mi.names),
                    check="column_ordered",
                    reason_code=SchemaErrorReason.COLUMN_NOT_ORDERED,
                )
            for position, idx_schema in enumerate(schema.indexes):
                mapping.append((position, idx_schema))
                used_levels.add(position)
        else:
            # Unordered – match by name first, then fallback to unused levels.
            for idx_schema in schema.indexes:
                if idx_schema.name is not None:
                    # Find the first unused level with this name
                    candidate_levels = [
                        i for i, n in enumerate(mi.names) if n == idx_schema.name and i not in used_levels
                    ]
                    if not candidate_levels:
                        raise SchemaError(
                            schema=schema,
                            data=mi,
                            message=f"index level with name '{idx_schema.name}' not found",
                            failure_cases=idx_schema.name,
                            check="column_in_index",
                            reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        )
                    level_pos = candidate_levels[0]
                else:
                    # Name is None – take the next unused level.
                    remaining = [i for i in range(mi.nlevels) if i not in used_levels]
                    if not remaining:
                        raise SchemaError(
                            schema=schema,
                            data=mi,
                            message="Ran out of index levels to map to unnamed schema component",
                            failure_cases=str(mi.names),
                            check="column_in_index",
                            reason_code=SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
                        )
                    level_pos = remaining[0]
                mapping.append((level_pos, idx_schema))
                used_levels.add(level_pos)
        return mapping

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
