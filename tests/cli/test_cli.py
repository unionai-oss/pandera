"""Tests for ``pandera`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

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


def test_infer_cli_yaml_roundtrip_validate(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    schema_path = tmp_path / "inferred.yaml"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(
        data_path,
        index=False,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "infer",
            "-d",
            str(data_path),
            "-o",
            str(schema_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert schema_path.is_file()
    assert (
        yaml.safe_load(schema_path.read_text(encoding="utf-8"))["schema_type"]
        == "dataframe"
    )

    proc2 = subprocess.run(
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
    assert proc2.returncode == 0, proc2.stderr
    assert "Validation succeeded" in proc2.stdout


def test_infer_cli_json(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "schema.json"
    pd.DataFrame({"id": [1]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "infer",
            "-d",
            str(data_path),
            "-o",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_type"] == "dataframe"
    assert "id" in payload["columns"]


def test_infer_cli_py_emits_script(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "schema.py"
    pd.DataFrame({"n": [1.0, 2.0]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "infer",
            "-d",
            str(data_path),
            "-o",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    text = out_path.read_text(encoding="utf-8")
    assert "DataFrameSchema" in text
    assert "Column" in text


def test_infer_cli_unknown_output_suffix_exits(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "schema.txt"
    pd.DataFrame({"a": [1]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "infer",
            "-d",
            str(data_path),
            "-o",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Could not infer output format" in (proc.stderr + proc.stdout)


def test_infer_cli_format_override_suffix(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "out.txt"
    pd.DataFrame({"a": [1]}).to_csv(data_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "infer",
            "-d",
            str(data_path),
            "-o",
            str(out_path),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_type"] == "dataframe"


def test_generate_cli_pandas_csv(tmp_path: Path):
    pytest.importorskip("hypothesis")
    schema_path = tmp_path / "schema.yaml"
    out_path = tmp_path / "sample.csv"
    schema = pa.DataFrameSchema({"a": pa.Column(int), "b": pa.Column(str)})
    io.to_yaml(schema, schema_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "generate",
            "-s",
            str(schema_path),
            "-o",
            str(out_path),
            "--size",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    df = pd.read_csv(out_path)
    assert len(df) == 4
    assert list(df.columns) == ["a", "b"]


def _netcdf_engine_available() -> bool:
    """xarray needs netCDF4, h5netcdf, or scipy to write netCDF files."""
    for engine in ("netCDF4", "h5netcdf", "scipy"):
        try:
            __import__(engine)
            return True
        except ImportError:
            continue
    return False


def test_generate_cli_xarray_netcdf(tmp_path: Path):
    pytest.importorskip("hypothesis")
    pytest.importorskip("xarray")
    if not _netcdf_engine_available():
        pytest.skip("no netCDF engine (netCDF4/h5netcdf/scipy) installed")
    from pandera.api.xarray.components import Coordinate, DataVar
    from pandera.api.xarray.container import DatasetSchema
    from pandera.io import xarray_io as xio

    schema_path = tmp_path / "ds.yaml"
    out_path = tmp_path / "out.nc"
    ds_schema = DatasetSchema(
        data_vars={"v": DataVar(dtype="float64", dims=("x",))},
        coords={"x": Coordinate(dtype="float64")},
        sizes={"x": 3},
    )
    xio.to_yaml(ds_schema, schema_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pandera",
            "generate",
            "-s",
            str(schema_path),
            "-o",
            str(out_path),
            "--size",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    import xarray as xr

    ds = xr.open_dataset(out_path)
    try:
        assert "v" in ds.data_vars
        assert ds.sizes["x"] == 3
    finally:
        ds.close()
