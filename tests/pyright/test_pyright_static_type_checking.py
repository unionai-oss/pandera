# pylint: skip-file
"""Unit tests for Pyright static type checking of pandera DataFrame types.

This module runs Pyright on the modules in ``tests/pyright/modules`` and
asserts expected diagnostics.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

test_module_dir = Path(__file__).parent
modules_dir = test_module_dir / "modules"


def _pyright_available() -> bool:
    if shutil.which("pyright") is not None:
        return True
    try:
        subprocess.run(
            [sys.executable, "-m", "pyright", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(
    not _pyright_available(),
    reason="pyright is not installed",
)


def _pyright_command(module_path: Path) -> list[str]:
    if shutil.which("pyright") is not None:
        return ["pyright", "--outputjson", str(module_path)]
    return [sys.executable, "-m", "pyright", "--outputjson", str(module_path)]


def _run_pyright(module_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        _pyright_command(module_path),
        cwd=str(test_module_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            "Pyright failed to run:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _filter_diagnostics(
    diagnostics: list[dict[str, Any]],
    *,
    module_name: str,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for diagnostic in diagnostics:
        file_path = Path(diagnostic.get("file", "")).name
        if file_path != module_name:
            continue
        if severity is not None and diagnostic.get("severity") != severity:
            continue
        filtered.append(diagnostic)
    return filtered


def _information_messages(
    diagnostics: list[dict[str, Any]], module_name: str
) -> list[str]:
    return [
        diagnostic["message"]
        for diagnostic in _filter_diagnostics(
            diagnostics, module_name=module_name, severity="information"
        )
    ]


@pytest.mark.parametrize(
    "module,expected_error_count,expected_rules,expected_type_messages",
    [
        (
            "schema_retention.py",
            0,
            [],
            [
                'Type of "result_check_types" is "DataFrame[InputSchema]"',
                'Type of "result_validate" is "DataFrame[InputSchema]"',
                'Type of "result_constructor" is "DataFrame[InputSchema]"',
                'Type of "result_generic" is "DataFrame[InputSchema]"',
            ],
        ),
        (
            "check_types_output.py",
            0,
            [],
            [
                'Type of "result" is "DataFrame[OutputSchema]"',
                'Type of "validated" is "DataFrame[InputSchema]"',
                'Type of "typed_df" is "DataFrame[InputSchema]"',
            ],
        ),
        (
            "typed_wrapper.py",
            0,
            [],
            [
                'Type of "df["year"]" is "Series[int]"',
            ],
        ),
        (
            "column_access.py",
            0,
            [],
            [
                'Type of "df["year"]" is "Unknown"',
                'Type of "df.year" is "Unknown"',
            ],
        ),
    ],
)
def test_pyright_static_typing_modules(
    module: str,
    expected_error_count: int,
    expected_rules: list[str],
    expected_type_messages: list[str],
) -> None:
    """Pyright diagnostics for static typing test modules."""
    module_path = modules_dir / module
    output = _run_pyright(module_path)
    diagnostics = output.get("generalDiagnostics", [])

    errors = _filter_diagnostics(
        diagnostics, module_name=module, severity="error"
    )
    assert len(errors) == expected_error_count
    assert [error.get("rule") for error in errors] == expected_rules

    information = _information_messages(diagnostics, module)
    for expected_message in expected_type_messages:
        assert any(expected_message in message for message in information), (
            f"Expected pyright information diagnostic containing "
            f"{expected_message!r}, got: {information}"
        )


@pytest.mark.parametrize(
    "module",
    [
        "schema_retention",
        "check_types_output",
        "typed_wrapper",
        "column_access",
    ],
)
def test_pyright_modules_importable(module: str) -> None:
    """Ensure pyright test modules are importable at runtime."""
    importlib.import_module(f"tests.pyright.modules.{module}")
