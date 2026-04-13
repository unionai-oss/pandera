"""Handle schema errors for Narwhals backends."""

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import ErrorHandler as _ErrorHandler


class ErrorHandler(_ErrorHandler):
    """Handler for schema- and data-level errors during Narwhals validation."""

    @staticmethod
    def _count_failure_cases(failure_cases) -> int:
        # failure_cases is always native at SchemaError boundary (Phase 6 contract).
        # nw.from_native wraps pl.DataFrame, pl.LazyFrame, and ibis.Table uniformly.
        # Python scalars (bool False, None, strings) raise TypeError — fall back to 0/1.
        # The isinstance(failure_cases, str) guard is removed: nw.from_native("string")
        # also raises TypeError, so the except branch returns 1 (same behavior, no guard needed).
        try:
            return int(
                nw.from_native(failure_cases, eager_only=False)
                .lazy()
                .select(nw.len())
                .collect()["len"][0]
            )
        except TypeError:
            return 0 if failure_cases is None else 1
