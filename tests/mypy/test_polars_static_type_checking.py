"""Test mypy static type checking for Polars DataFrameModel.

This module uses subprocess and pytest to capture the output
of calling mypy on the python modules in the tests/mypy/polars_modules folder.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

test_module_dir = Path(os.path.dirname(__file__))


def _get_mypy_errors(
    module_name: str,
    stdout,
):
    """Parse mypy output and return list of errors."""
    errors = []
    for line in stdout.split("\n"):
        if module_name in line and "error:" in line:
            errors.append(line)
    return errors


POLARS_MODEL_COLUMN_ATTRS_ERRORS = [
    'Argument 1 to "print_string" has incompatible type "int"; expected "str"',
    'Argument 1 to "print_string" has incompatible type "float"; expected "str"',
]


@pytest.mark.parametrize(
    ["module", "config", "expected_errors"],
    [
        [
            "polars_model_column_attrs.py",
            "no_plugin.ini",
            POLARS_MODEL_COLUMN_ATTRS_ERRORS,
        ],
        ["polars_model_column_attrs.py", "plugin_mypy.ini", []],
    ],
)
def test_polars_mypy_typing(capfd, module, config, expected_errors) -> None:
    """Test that mypy plugin correctly handles Polars DataFrameModel field types."""
    cache_dir = str(
        test_module_dir
        / ".mypy_cache"
        / f"{module.replace('.', '_')}-{config.replace('.', '_')}"
    )
    commands = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(test_module_dir / "config" / config),
        str(test_module_dir / "polars_modules" / module),
        "--cache-dir",
        cache_dir,
        "--show-error-codes",
    ]
    # pylint: disable=subprocess-run-check
    result = subprocess.run(commands, text=True)
    # NOTE: mypy return code is 0 if no errors were found, 1 if errors were found
    # or 2 if there was a failure in checking
    assert result.returncode in (0, 1)
    resulting_errors = _get_mypy_errors(module, capfd.readouterr().out)

    assert len(expected_errors) == len(resulting_errors), (
        f"Expected {len(expected_errors)} errors but got {len(resulting_errors)}. "
        f"Errors: {resulting_errors}"
    )
    for expected, error in zip(expected_errors, resulting_errors):
        assert expected in error, f"Expected '{expected}' in error '{error}'"


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
