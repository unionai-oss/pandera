"""Polars Parsing, Validation, and Error Reporting Backends."""

import polars as pl

from collections import defaultdict
from typing import List

from pandera.backends.base import BaseSchemaBackend
from pandera.errors import SchemaError, FailureCaseMetadata


class PolarsSchemaBackend(BaseSchemaBackend):
    def failure_cases_metadata(
        self,
        schema_name: str,
        schema_errors: List[SchemaError],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception."""

        scalar_failure_cases = defaultdict(list)
        error_counts = defaultdict(int)

        for err in schema_errors:

            error_counts[err.reason_code] += 1

            check_identifier = (
                None
                if err.check is None
                else err.check
                if isinstance(err.check, str)
                else err.check.error
                if err.check.error is not None
                else err.check.name
                if err.check.name is not None
                else str(err.check)
            )

            if isinstance(err.failure_cases, pl.LazyFrame):
                raise NotImplementedError

            scalar_failure_cases["schema_context"].append(
                err.schema.__class__.__name__
            )
            scalar_failure_cases["column"].append(err.schema.name)
            scalar_failure_cases["check"].append(check_identifier)
            scalar_failure_cases["check_number"].append(err.check_index)
            scalar_failure_cases["failure_case"].append(err.failure_cases)
            scalar_failure_cases["index"].append(None)

        failure_cases = pl.DataFrame(scalar_failure_cases)

        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=FAILURE_CASE_TEMPLATE.format(
                schema_name=schema_name,
                error_count=sum(error_counts.values()),
                failure_cases=str(failure_cases),
            ),
            error_counts=error_counts,
        )


FAILURE_CASE_TEMPLATE = """
Schema {schema_name}: A total of {error_count} errors were found.

{failure_cases}
""".strip()
