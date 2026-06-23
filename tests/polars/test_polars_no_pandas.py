"""Regression tests for using ``pandera[polars]`` without numpy/pandas.

The polars extra is disjoint from the pandas extra in ``pyproject.toml``, so the
documented ``pip install 'pandera[polars]'`` may run where neither numpy nor
pandas is importable. Builtin check registration must not depend on importing
``pandera.api.pandas.types`` (which imports numpy/pandas at module load). See
GH #2387.

Like ``tests/dask/test_dask_not_installed.py`` this blocks the missing modules
in-process and re-imports pandera under the block. A meta-path finder is used
rather than the ``sys.modules[name] = None`` sentinel because
``pandera/__init__.py`` only swallows an ImportError whose message starts with
``No module named 'numpy'``; the sentinel raises a differently-worded message
that would be re-raised.
"""

import importlib
import importlib.abc
import sys

import pytest

pytest.importorskip("polars")

import polars as pl

from pandera.backends.register_checks import _load_get_backend_types_from_mro


class _BlockNumpyPandas(importlib.abc.MetaPathFinder):
    """Meta-path finder that makes numpy and pandas look uninstalled."""

    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in ("numpy", "pandas"):
            raise ModuleNotFoundError(f"No module named {root!r}")
        return None


def _drop_cached():
    for name in list(sys.modules):
        if name.split(".", 1)[0] in ("pandera", "numpy", "pandas"):
            del sys.modules[name]


@pytest.fixture
def without_numpy_pandas():
    """Make numpy/pandas unimportable and re-import pandera under the block.

    Restores the module table and the backend-loader cache afterwards so the
    block does not leak into the rest of the session, which has pandas.
    """
    saved = dict(sys.modules)
    finder = _BlockNumpyPandas()
    sys.meta_path.insert(0, finder)
    _drop_cached()
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        _drop_cached()
        sys.modules.update(saved)
        # The loader caches None while pandas is blocked; reset it so later
        # tests re-import the pandas backend types.
        _load_get_backend_types_from_mro.cache_clear()


@pytest.mark.usefixtures("without_numpy_pandas")
def test_builtin_checks_register_without_numpy_pandas():
    """Builtin polars checks register when numpy/pandas are not importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pandas")

    import pandera.polars as pa
    from pandera.polars import Column, DataFrameSchema

    # Constructing builtin checks must not raise KeyError (GH #2387).
    DataFrameSchema({"a": Column(pl.Int64, pa.Check.ge(0))})
    DataFrameSchema({"a": Column(pl.Utf8, pa.Check.isin(["x", "y"]))})


@pytest.mark.usefixtures("without_numpy_pandas")
def test_failing_check_reports_value_without_numpy_pandas():
    """A failing builtin check reports the value, not a swallowed import error."""
    import pandera.polars as pa
    from pandera.polars import Column, DataFrameSchema

    schema = DataFrameSchema({"a": Column(pl.Int64, pa.Check.ge(0))})
    with pytest.raises(
        pa.errors.SchemaError, match="greater_than_or_equal_to"
    ):
        schema.validate(pl.DataFrame({"a": [-1]}))
