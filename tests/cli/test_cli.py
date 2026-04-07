"""Tests for ``pandera`` CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import pandera as pa
from pandera.io import pandas_io as io


def test_validate_cli_pandas_csv_ok(tmp_path: Path):
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    io.to_yaml(schema, schema_path)
    pd.DataFrame({"a": [1, 2]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "validate",
            "-s",
            str(schema_path),
            "-d",
            str(data_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Validation succeeded" in proc.stdout


def test_validate_cli_pandas_csv_fail(tmp_path: Path):
    schema = pa.DataFrameSchema({"a": pa.Column(int, pa.Check.ge(0))})
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "bad.csv"
    io.to_yaml(schema, schema_path)
    pd.DataFrame({"a": [-1]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "validate",
            "-s",
            str(schema_path),
            "-d",
            str(data_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert (
        "Validation failed" in proc.stderr
        or "Validation failed" in proc.stdout
    )


@pytest.mark.parametrize("backend_flag", ["pandas", None])
def test_validate_cli_explicit_backend_matches(tmp_path: Path, backend_flag):
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    io.to_yaml(schema, schema_path)
    pd.DataFrame({"a": [1]}).to_csv(data_path, index=False)

    cmd = [
        sys.executable,
        "-m",
        "pandera",
        "validate",
        "-s",
        str(schema_path),
        "-d",
        str(data_path),
    ]
    if backend_flag:
        cmd.extend(["--backend", backend_flag])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_validate_cli_backend_mismatch_exits(tmp_path: Path):
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    io.to_yaml(schema, schema_path)
    pd.DataFrame({"a": [1]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "validate",
            "-s",
            str(schema_path),
            "-d",
            str(data_path),
            "--backend",
            "polars",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "does not match schema" in (proc.stderr + proc.stdout)
