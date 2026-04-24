"""DataTree schema backend."""

from __future__ import annotations

from typing import Any

from pandera.api.base.error_handler import ErrorHandler, get_error_category
from pandera.api.xarray.container import DatasetSchema, DataTreeSchema
from pandera.backends.base import CoreCheckResult
from pandera.backends.xarray.base import XarraySchemaBackend
from pandera.backends.xarray.container import DatasetSchemaBackend
from pandera.config import ValidationScope
from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors
from pandera.validation_depth import validate_scope


def _collect(
    error_handler: ErrorHandler,
    schema: Any,
    check_obj: Any,
    results: list[CoreCheckResult],
) -> None:
    for result in results:
        if result.passed:
            continue
        err = result.schema_error or SchemaError(
            schema,
            data=check_obj,
            message=result.message or "",
            failure_cases=result.failure_cases,
            check=result.check,
            check_index=result.check_index,
            check_output=result.check_output,
            reason_code=result.reason_code,
        )
        assert result.reason_code is not None
        error_handler.collect_error(
            get_error_category(result.reason_code),
            result.reason_code,
            err,
            original_exc=result.original_exc,
        )


class DataTreeSchemaBackend(XarraySchemaBackend):
    """Validate :class:`~xarray.DataTree` against
    :class:`DataTreeSchema`."""

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_attrs(
        self, check_obj, schema: DataTreeSchema
    ) -> list[CoreCheckResult]:
        results: list[CoreCheckResult] = []
        if not schema.attrs:
            return results
        for ak, av in schema.attrs.items():
            if ak not in check_obj.attrs or check_obj.attrs[ak] != av:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="attrs",
                        reason_code=(SchemaErrorReason.SCHEMA_COMPONENT_CHECK),
                        message=(
                            f"attribute mismatch {ak!r}: "
                            f"expected {av!r}, "
                            f"got {check_obj.attrs.get(ak)!r}"
                        ),
                        failure_cases=str(check_obj.attrs.get(ak)),
                    )
                )
        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_strict_children(
        self, check_obj, schema: DataTreeSchema
    ) -> list[CoreCheckResult]:
        results: list[CoreCheckResult] = []
        if not schema.strict:
            return results
        expected = set()
        for key in schema.children:
            parts = key.strip("/").split("/")
            expected.add(parts[0])
        for child_name in check_obj.children:
            if child_name not in expected:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="strict_children",
                        reason_code=(SchemaErrorReason.COLUMN_NOT_IN_SCHEMA),
                        message=(f"unexpected child node {child_name!r}"),
                        failure_cases=child_name,
                    )
                )
        return results

    def _resolve_node(self, tree, path: str):
        """Resolve a path to a DataTree node."""
        path = path.strip("/")
        if not path:
            return tree
        try:
            return tree[path]
        except KeyError:
            return None

    def _validate_child(
        self,
        tree,
        path: str,
        child_schema,
        error_handler: ErrorHandler,
        *,
        head: int | None,
        tail: int | None,
        sample: int | None,
        random_state: int | None,
        lazy: bool,
    ) -> list[CoreCheckResult]:
        results: list[CoreCheckResult] = []
        node = self._resolve_node(tree, path)
        if node is None:
            results.append(
                CoreCheckResult(
                    passed=False,
                    check=f"child[{path!r}]",
                    reason_code=(SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME),
                    message=(f"missing child node at path {path!r}"),
                    failure_cases=path,
                )
            )
            return results

        if isinstance(child_schema, DataTreeSchema):
            try:
                self.validate(
                    node,
                    child_schema,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except SchemaErrors as exc:
                for schema_err in exc.schema_errors:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check=f"child[{path!r}]",
                            reason_code=schema_err.reason_code,
                            schema_error=SchemaError(
                                child_schema,
                                data=node,
                                message=(
                                    f"DataTreeSchema failed "
                                    f"at path {path!r}: "
                                    f"{schema_err.args[0] if schema_err.args else ''}"
                                ),
                                reason_code=schema_err.reason_code,
                            ),
                        )
                    )
            except SchemaError as e:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check=f"child[{path!r}]",
                        reason_code=e.reason_code,
                        schema_error=SchemaError(
                            child_schema,
                            data=node,
                            message=(
                                f"DataTreeSchema failed "
                                f"at path {path!r}: "
                                f"{e.args[0] if e.args else ''}"
                            ),
                            reason_code=e.reason_code,
                        ),
                    )
                )
        elif isinstance(child_schema, DatasetSchema):
            ds_backend = DatasetSchemaBackend()
            node_ds = node.to_dataset(inherit=False)
            try:
                ds_backend.validate(
                    node_ds,
                    child_schema,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except SchemaErrors as exc:
                for schema_err in exc.schema_errors:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check=f"child[{path!r}]",
                            reason_code=schema_err.reason_code,
                            schema_error=SchemaError(
                                child_schema,
                                data=node_ds,
                                message=(
                                    f"DataTreeSchema failed "
                                    f"at path {path!r}: "
                                    f"{schema_err.args[0] if schema_err.args else ''}"
                                ),
                                reason_code=schema_err.reason_code,
                            ),
                        )
                    )
            except SchemaError as e:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check=f"child[{path!r}]",
                        reason_code=e.reason_code,
                        schema_error=SchemaError(
                            child_schema,
                            data=node_ds,
                            message=(
                                f"DataTreeSchema failed "
                                f"at path {path!r}: "
                                f"{e.args[0] if e.args else ''}"
                            ),
                            reason_code=e.reason_code,
                        ),
                    )
                )
        return results

    def validate(
        self,
        check_obj,
        schema: DataTreeSchema,
        *,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ):
        error_handler = ErrorHandler(lazy)

        results: list[CoreCheckResult] = []

        results.extend(self.check_attrs(check_obj, schema))

        if schema.dataset is not None:
            ds_backend = DatasetSchemaBackend()
            node_ds = check_obj.to_dataset(inherit=False)
            try:
                ds_backend.validate(
                    node_ds,
                    schema.dataset,
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                )
            except SchemaErrors as exc:
                for e in exc.schema_errors:
                    results.append(
                        CoreCheckResult(
                            passed=False,
                            check="node_dataset",
                            reason_code=e.reason_code,
                            schema_error=e,
                        )
                    )
            except SchemaError as e:
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check="node_dataset",
                        reason_code=e.reason_code,
                        schema_error=e,
                    )
                )

        for path, child_schema in schema.children.items():
            child_results = self._validate_child(
                check_obj,
                path,
                child_schema,
                error_handler,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
            )
            results.extend(child_results)

        results.extend(self.check_strict_children(check_obj, schema))

        _collect(error_handler, schema, check_obj, results)

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )
        return check_obj
