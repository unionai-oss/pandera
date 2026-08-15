"""Backend registration tests for the pyarrow API."""

import narwhals.stable.v1 as nw
import pyarrow
import pytest

import pandera.pyarrow as pa
from pandera.api.checks import Check
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.backends.narwhals.components import ColumnBackend
from pandera.backends.narwhals.container import DataFrameSchemaBackend
from pandera.backends.pyarrow.register import register_pyarrow_backends


def test_pyarrow_table_resolves_narwhals_backends():
    register_pyarrow_backends()
    tbl = pyarrow.table({"a": [1]})
    assert isinstance(
        pa.DataFrameSchema.get_backend(tbl), DataFrameSchemaBackend
    )
    assert isinstance(pa.Column.get_backend(tbl), ColumnBackend)
    assert Check.get_backend(tbl) is NarwhalsCheckBackend


def test_importing_api_module_first_does_not_deadlock():
    """``pandera.api.pyarrow.*`` must be importable without the entry point.

    Registering eagerly from ``pandera.backends.pyarrow.__init__`` created a
    circular import: the container imports the register module, whose package
    ``__init__`` imported the container back.
    """
    import subprocess
    import sys

    for module in (
        "pandera.api.pyarrow.container",
        "pandera.api.pyarrow.components",
        "pandera.api.pyarrow.model",
    ):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


def test_narwhals_check_dispatch_does_not_register_unrelated_backends():
    """A check on a narwhals frame must not pull in every backend.

    ``register_default_check_backends`` used to eagerly register the polars,
    ibis and pyspark backends for any ``narwhals.*`` check object. That primed
    each library's registration ``lru_cache``, so a later wipe of the shared
    ``BACKEND_REGISTRY`` left them permanently unregistered.
    """
    ibis_register = pytest.importorskip(
        "pandera.backends.ibis.register"
    ).register_ibis_backends

    register_pyarrow_backends()
    # Ensure the narwhals check backends are present so the fast path applies.
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend, force=True)
    ibis_register.cache_clear()

    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.gt(0))})
    schema.validate(pyarrow.table({"a": [1, 2]}))

    assert ibis_register.cache_info().currsize == 0
