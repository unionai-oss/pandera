"""Tests for the scoped numpy 2 alias patch used for pyspark < 4.

See https://github.com/unionai-oss/pandera/issues/2134
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pandera import _patch_numpy2 as patch_module

NUMPY2_REMOVED_ALIASES = ("float_", "NaN", "string_", "unicode_")

requires_numpy2 = pytest.mark.skipif(
    int(np.__version__.split(".", maxsplit=1)[0]) < 2,
    reason="numpy 1.x still ships the aliases natively",
)


def _run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter so numpy's namespace is pristine."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "pyspark_version, expected",
    [
        ("3.2.0", True),
        ("3.5.1", True),
        ("3.5.0.dev0", True),
        ("4.0.0", False),
        ("4.0.0.dev1", False),
        ("4.1.0rc1", False),
        ("10.0.0", False),
        (None, False),
        ("not-a-version", False),
    ],
)
def test_pyspark_needs_numpy2_patch(monkeypatch, pyspark_version, expected):
    """Only pyspark < 4 requires the numpy alias patch."""
    monkeypatch.setattr(
        patch_module, "_get_pyspark_version", lambda: pyspark_version
    )
    assert patch_module._pyspark_needs_numpy2_patch() is expected


def test_get_pyspark_version_when_not_installed(monkeypatch):
    """A missing pyspark distribution maps to ``None``."""

    def _raise(_name):
        raise patch_module.metadata.PackageNotFoundError("pyspark")

    monkeypatch.setattr(patch_module.metadata, "version", _raise)
    assert patch_module._get_pyspark_version() is None


@requires_numpy2
@pytest.mark.parametrize(
    "pyspark_version, patched",
    [("3.5.1", True), ("4.0.0", False), (None, False)],
)
def test_patch_numpy2_is_gated_on_pyspark_version(pyspark_version, patched):
    """numpy is only modified when the pyspark version requires it."""
    result = _run_python(
        f"""
        import importlib.util
        import os

        import numpy as np

        # Load the module standalone: ``import pandera`` would import the
        # pandas API, which applies the patch for a real pyspark < 4 install.
        pandera_dir = os.path.dirname(importlib.util.find_spec("pandera").origin)
        spec = importlib.util.spec_from_file_location(
            "_patch_numpy2", os.path.join(pandera_dir, "_patch_numpy2.py")
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        m._get_pyspark_version = lambda: {pyspark_version!r}
        m._patch_numpy2()
        m._patch_numpy2()  # idempotent
        present = [a for a in {NUMPY2_REMOVED_ALIASES!r} if hasattr(np, a)]
        print(",".join(present))
        if present:
            assert np.float_ is np.float64
            assert np.string_ is np.bytes_
            assert np.unicode_ is np.str_
            assert np.isnan(np.NaN)
        """
    )
    assert result.returncode == 0, result.stderr
    present = [a for a in result.stdout.strip().split(",") if a]
    expected = list(NUMPY2_REMOVED_ALIASES) if patched else []
    assert present == expected


@requires_numpy2
@pytest.mark.skipif(
    patch_module._pyspark_needs_numpy2_patch(),
    reason="pyspark < 4 is installed; pandera.pandas imports pyspark.pandas "
    "eagerly, which still requires the numpy alias patch",
)
@pytest.mark.parametrize("module", ["pandera.pandas", "pandera"])
def test_pandas_import_does_not_patch_numpy(module):
    """Importing pandera's pandas API must not re-add removed numpy aliases."""
    pytest.importorskip("pandas")
    result = _run_python(
        f"""
        import numpy as np
        import {module}

        present = [a for a in {NUMPY2_REMOVED_ALIASES!r} if hasattr(np, a)]
        assert not present, present
        try:
            np.float_
        except AttributeError:
            pass
        else:
            raise AssertionError("np.float_ should raise on numpy 2")
        """
    )
    assert result.returncode == 0, result.stderr
