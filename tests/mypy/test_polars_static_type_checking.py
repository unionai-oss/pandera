"""Unit tests for static type checking of polars schemas."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

test_module_dir = Path(os.path.dirname(__file__))


def test_mypy_polars_column_parametrized_dtypes() -> None:
    """Check that Column typing accepts parametrized polars dtypes."""
    pytest.importorskip("polars")

    cache_dir = str(test_module_dir / ".mypy_cache" / "test-polars-dtypes")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(test_module_dir / "polars_modules" / "polars_dtypes.py"),
            "--cache-dir",
            cache_dir,
            "--config-file",
            str(test_module_dir / "config" / "no_plugin.ini"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
