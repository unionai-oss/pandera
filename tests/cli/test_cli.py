"""Tests for ``pandera`` CLI."""

from __future__ import annotations

import importlib.util
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
    assert "does not match the schema's api 'pandas'" in (
        proc.stderr + proc.stdout
    )


def test_validate_cli_api_field_mismatch_exits(tmp_path: Path):
    """A hand-edited ``api`` field contradicting ``schema_type`` is rejected."""
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    payload = io.serialize_schema(schema)
    payload["api"] = "polars"
    schema_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "api 'polars' does not match schema_type 'dataframe'" in (
        proc.stderr + proc.stdout
    )


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
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert payload["schema_type"] == "dataframe"
    assert payload["api"] == "pandas"

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
    assert payload["api"] == "pandas"
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


# --- infer --format py: the emitted script must actually run ---


def _load_emitted_module(path: Path):
    """Import an emitted schema/model script the way a user would.

    ``DataFrameModel.to_schema`` resolves annotations through the defining
    module's globals, so the script has to be imported as a module rather
    than ``exec``-ed into a bare dict.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _run_infer_script(
    tmp_path: Path,
    *,
    backend: str,
    script_type: str,
) -> Path:
    """``pandera infer`` to a .py file; return the emitted script path."""
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / f"schema_{backend.replace('.', '_')}.py"
    pd.DataFrame(
        {
            "n": [1, 2],
            "s": ["ab", "cde"],
            "f": [1.5, 2.5],
        }
    ).to_csv(data_path, index=False)

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
            "--backend",
            backend,
            "--script-type",
            script_type,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return out_path


@pytest.mark.parametrize("backend", ["pandas", "polars", "ibis"])
def test_infer_py_schema_script_is_executable(tmp_path: Path, backend: str):
    """The emitted ``DataFrameSchema`` script imports and runs.

    Dtype literals are engine-specific, so a script emitted for one backend
    is only valid if it uses that backend's dtype expressions.
    """
    pytest.importorskip(backend.split(".")[0])
    path = _run_infer_script(tmp_path, backend=backend, script_type="schema")
    module = _load_emitted_module(path)
    assert set(module.schema.columns) == {"n", "s", "f"}


@pytest.mark.parametrize("backend", ["pandas", "polars", "ibis"])
def test_infer_py_model_script_is_executable(tmp_path: Path, backend: str):
    """The emitted ``DataFrameModel`` script imports and runs."""
    pytest.importorskip(backend.split(".")[0])
    path = _run_infer_script(tmp_path, backend=backend, script_type="model")
    module = _load_emitted_module(path)
    assert set(module.GeneratedModel.to_schema().columns) == {"n", "s", "f"}


def test_infer_py_script_handles_non_identifier_columns(tmp_path: Path):
    """Column labels that aren't Python identifiers become aliased fields."""
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "model.py"
    pd.DataFrame({"first name": [1, 2], "class": [3, 4]}).to_csv(
        data_path, index=False
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
            str(out_path),
            "--script-type",
            "model",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    module = _load_emitted_module(out_path)
    schema = module.GeneratedModel.to_schema()
    assert set(schema.columns) == {"first name", "class"}


def test_infer_rejects_narwhals_backend(tmp_path: Path):
    """``narwhals`` is a validation backend, not a dataframe API to infer."""
    data_path = tmp_path / "data.csv"
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
            str(tmp_path / "schema.yaml"),
            "--backend",
            "narwhals",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "cannot be used with `infer`" in (proc.stderr + proc.stdout)


@pytest.mark.parametrize(
    "value",
    ["carriage\rreturn", "new\nline", 'embedded "quote"', "comma,separated"],
)
def test_generate_csv_writer_round_trips_special_characters(
    tmp_path: Path, value: str
):
    """Synthetic strings are arbitrary text; CSV output must survive it.

    A bare ``\\r`` written unquoted is treated as a row break by readers,
    silently splitting one record into two.
    """
    from pandera._cli.generate import _write_generated_tabular_pandas

    out_path = tmp_path / "out.csv"
    df = pd.DataFrame({"i": [1], "s": [value]})
    _write_generated_tabular_pandas(df, out_path, "csv")

    back = pd.read_csv(out_path)
    assert back["s"].tolist() == [value]
    assert back["i"].tolist() == [1]


def test_generate_csv_writer_handles_nul_character(tmp_path: Path):
    """Hypothesis draws NUL, which CSV cannot represent.

    On Python < 3.12 the csv writer stores an unset ``escapechar`` as 0, so a
    NUL in the data compares equal to it and ``to_csv`` fails outright with
    "need to escape, but no escapechar set" in every quoting mode.
    """
    from pandera._cli.generate import _write_generated_tabular_pandas

    out_path = tmp_path / "out.csv"
    df = pd.DataFrame({"i": [1], "s": ["has\x00nul"]})
    _write_generated_tabular_pandas(df, out_path, "csv")

    back = pd.read_csv(out_path)
    assert len(back) == 1
    written = back["s"].iloc[0]
    assert "\x00" not in written
    # length is preserved so inferred str_length checks still hold
    assert len(written) == len("has\x00nul")


def test_generate_csv_writer_does_not_mutate_input(tmp_path: Path):
    """Sanitizing for CSV must not alter the caller's DataFrame."""
    from pandera._cli.generate import _write_generated_tabular_pandas

    df = pd.DataFrame({"s": ["has\x00nul"]})
    _write_generated_tabular_pandas(df, tmp_path / "out.csv", "csv")
    assert df["s"].iloc[0] == "has\x00nul"


@pytest.mark.parametrize("writer_key", ["json", "parquet", "feather"])
def test_generate_non_csv_writers_preserve_nul(
    tmp_path: Path, writer_key: str
):
    """Only CSV needs the substitution; binary/JSON formats carry NUL fine."""
    from pandera._cli.generate import _write_generated_tabular_pandas

    out_path = tmp_path / f"out.{writer_key}"
    df = pd.DataFrame({"s": ["has\x00nul"]})
    _write_generated_tabular_pandas(df, out_path, writer_key)

    reader = {
        "json": pd.read_json,
        "parquet": pd.read_parquet,
        "feather": pd.read_feather,
    }[writer_key]
    assert reader(out_path)["s"].iloc[0] == "has\x00nul"
