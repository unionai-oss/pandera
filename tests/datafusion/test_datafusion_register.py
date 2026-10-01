"""Backend registration tests for the DataFusion API."""

import subprocess
import sys
from unittest.mock import patch

import datafusion
import narwhals.stable.v1 as nw
import pytest

import pandera.datafusion as pa
from pandera.api.checks import Check
from pandera.backends.datafusion.register import register_datafusion_backends
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.backends.narwhals.components import ColumnBackend
from pandera.backends.narwhals.container import DataFrameSchemaBackend


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_datafusion_dataframe_resolves_narwhals_backends(make_df):
    register_datafusion_backends()
    df = make_df({"a": [1]})
    assert isinstance(df, datafusion.DataFrame)
    assert isinstance(
        pa.DataFrameSchema.get_backend(df), DataFrameSchemaBackend
    )
    assert isinstance(pa.Column.get_backend(df), ColumnBackend)
    assert isinstance(
        pa.Column.get_backend(check_type=datafusion.DataFrame), ColumnBackend
    )
    assert Check.get_backend(df) is NarwhalsCheckBackend


def test_narwhals_wraps_datafusion_via_plugin(make_df):
    """The plugin must route ``datafusion.DataFrame`` to a lazy frame."""
    wrapped = nw.from_native(
        make_df({"a": [1]}), eager_or_interchange_only=False
    )
    assert isinstance(wrapped, nw.LazyFrame)
    assert wrapped.implementation is nw.Implementation.UNKNOWN


def test_missing_plugin_raises_helpful_error():
    register_datafusion_backends.cache_clear()
    with patch.dict(sys.modules, {"narwhals_datafusion": None}):
        with pytest.raises(ImportError, match="narwhals-datafusion"):
            register_datafusion_backends()
    register_datafusion_backends.cache_clear()
    register_datafusion_backends()


def test_importing_api_module_first_does_not_deadlock():
    """``pandera.api.datafusion.*`` must be importable without the entry point."""
    for module in (
        "pandera.api.datafusion.container",
        "pandera.api.datafusion.components",
        "pandera.api.datafusion.model",
    ):
        proc = _run(f"import {module}")
        assert proc.returncode == 0, proc.stderr


def test_narwhals_check_dispatch_does_not_register_unrelated_backends(make_df):
    """Validating a DataFusion frame must not pull in the ibis backends."""
    ibis_register = pytest.importorskip(
        "pandera.backends.ibis.register"
    ).register_ibis_backends

    register_datafusion_backends()
    ibis_register.cache_clear()

    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    schema.validate(make_df({"a": [1, 2]}))

    assert ibis_register.cache_info().currsize == 0


@pytest.mark.xfail(
    reason="register_default_check_backends has no datafusion branch, so a "
    "check called before any schema validation finds no backend",
    strict=True,
)
def test_check_called_directly_in_fresh_process():
    proc = _run(
        "import pyarrow\n"
        "from datafusion import SessionContext\n"
        "import pandera.datafusion as pa\n"
        "df = SessionContext().from_arrow(pyarrow.table({'a': [1]}))\n"
        "pa.Check.gt(0)(df, 'a')\n"
    )
    assert proc.returncode == 0, proc.stderr
