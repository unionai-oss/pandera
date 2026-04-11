"""Shared fixtures for narwhals backend tests.

CI Matrix (TEST-01, TEST-02, TEST-03):

- tests/polars/ runs WITHOUT narwhals installed (CI job: unit-tests-dataframe-extras,
  extra=polars). Guarded by tests/polars/conftest.py which re-registers the native
  polars backend at session start (see TEST-01 in that file).
- tests/ibis/ runs WITHOUT narwhals installed (CI job: unit-tests-dataframe-extras,
  extra=ibis). Guarded by tests/ibis/conftest.py.
- tests/backends/narwhals/ (this directory) runs WITH narwhals + polars + ibis all
  installed together (CI job: unit-tests-narwhals, extra=narwhals). The
  `make_narwhals_frame` fixture below parametrizes every test across the three
  supported native frame types (pl.DataFrame, pl.LazyFrame, ibis.Table) so each
  test runs 3 times and no frame type is silently skipped (TEST-02).

See .github/workflows/ci-tests.yml for the full matrix and .planning/REQUIREMENTS.md
for TEST-01, TEST-02, and TEST-03 definitions.
"""
import warnings

import pytest
import polars as pl
import narwhals.stable.v1 as nw


@pytest.fixture(autouse=True, scope="module")
def _suppress_narwhals_warning():
    """Initialise narwhals backends and suppress the auto-activation UserWarning.

    Calls register_polars_backends() once per module so that:
    - builtin_checks side-effect runs (populates Dispatcher._function_registry)
    - NarwhalsCheckBackend, ColumnBackend, DataFrameSchemaBackend are registered
    - Tests that call NarwhalsCheckBackend directly do not need to trigger
      schema.validate() first.

    UserWarning is suppressed to keep test output clean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from pandera.backends.polars.register import register_polars_backends
        from pandera.backends.ibis.register import register_ibis_backends
        register_polars_backends.cache_clear()
        register_ibis_backends.cache_clear()
        register_polars_backends()
        register_ibis_backends()
        yield


@pytest.fixture(
    params=["polars_eager", "polars_lazy", "ibis_table"],
    ids=["polars_eager", "polars_lazy", "ibis_table"],
)
def make_narwhals_frame(request):
    """Return a callable that creates an nw frame across all 3 supported native types.

    TEST-02: parametrizes Narwhals backend tests across pl.DataFrame (eager),
    pl.LazyFrame (lazy), and ibis.Table — all three supported native frame types.
    """
    backend = request.param

    def _make(data: dict):
        if backend == "polars_eager":
            return nw.from_native(pl.DataFrame(data), eager_only=True)
        elif backend == "polars_lazy":
            return nw.from_native(
                pl.LazyFrame(data), eager_or_interchange_only=False
            )
        elif backend == "ibis_table":
            import pandas as pd
            import ibis
            return nw.from_native(
                ibis.memtable(pd.DataFrame(data)),
                eager_or_interchange_only=False,
            )

    return _make
