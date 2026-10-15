"""DataFusion unit test-specific configuration."""

import datafusion
import pyarrow
import pytest
from datafusion import SessionContext

from pandera.config import CONFIG, ValidationDepth, reset_config_context
from pandera.errors import SchemaError, SchemaErrors


@pytest.fixture(scope="function", autouse=True)
def validation_depth_schema_and_data():
    """Run data-level checks in the DataFusion unit tests.

    A ``datafusion.DataFrame`` is lazy, so the API defaults to
    ``SCHEMA_ONLY``; the tests opt into ``SCHEMA_AND_DATA`` the same way the
    ibis suite does.
    """
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth
        reset_config_context()


@pytest.fixture
def default_depth():
    """Use the backend's own default depth instead of the autouse one."""
    _validation_depth = CONFIG.validation_depth
    CONFIG.validation_depth = None
    try:
        yield
    finally:
        CONFIG.validation_depth = _validation_depth


@pytest.fixture(scope="session")
def ctx() -> SessionContext:
    return SessionContext()


@pytest.fixture
def make_df(ctx):
    """Build a ``datafusion.DataFrame`` from a column dict or pyarrow table."""

    def _make(data):
        if not isinstance(data, pyarrow.Table):
            data = pyarrow.table(data)
        return ctx.from_arrow(data)

    return _make


@pytest.fixture
def fails():
    """Validate, expect a failure, and return it after checking its reasons.

    ``schema`` is anything with a ``validate`` method: a schema, a column or
    a model.
    """

    def _fails(schema, df, *reasons, **validate_kwargs):
        with pytest.raises((SchemaError, SchemaErrors)) as exc_info:
            schema.validate(df, **validate_kwargs)
        error = exc_info.value
        errors = getattr(error, "schema_errors", [error])
        assert {err.reason_code for err in errors} == set(reasons)
        return error

    return _fails


@pytest.fixture
def executions(monkeypatch):
    """Row counts of every result DataFusion hands back during validation."""
    row_counts = []
    to_arrow_table = datafusion.DataFrame.to_arrow_table

    def recording(self):
        table = to_arrow_table(self)
        row_counts.append(table.num_rows)
        return table

    monkeypatch.setattr(datafusion.DataFrame, "to_arrow_table", recording)
    return row_counts
