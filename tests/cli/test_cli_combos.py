"""Parametrized CLI tests: schema file format × dataset format × backend.

``validate``: pandas API (``.yaml`` / ``.yml`` / ``.json`` × csv/parquet/json/feather
× implicit or explicit ``pandas``); Polars; Modin/Dask/pyspark.pandas library
tags; Ibis (csv/parquet); PySpark SQL (representative triple; needs
``PANDERA_RUN_SPARK_CLI=1`` or runs in the nox ``pyspark`` session).

``infer``: pandas and Polars data formats × YAML/JSON schema output; ``--format``
with extensionless paths.

``generate``: pandas schema YAML/JSON → CSV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from tests.cli.conftest import (
    assert_validate_ok,
    run_infer,
    run_validate,
    write_pandas_api_schema,
    write_pandas_compatible_data,
)

# Spark starts a JVM per ``python -m pandera`` subprocess; skip unless opted in
# for CI or machines with a working local Spark (see PySpark test workflow).
_SKIP_SPARK_CLI_COMBOS = os.environ.get("PANDERA_RUN_SPARK_CLI") != "1"

# --- validate: pandas API + default / explicit pandas backend ---


@pytest.mark.parametrize("schema_kind", ["yaml", "yml", "json"])
@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json", "feather"])
@pytest.mark.parametrize("backend_flag", [None, "pandas"])
def test_validate_pandas_api_all_schema_and_data_formats(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
    backend_flag: str | None,
) -> None:
    ext = {"yaml": ".yaml", "yml": ".yml", "json": ".json"}[schema_kind]
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    write_pandas_api_schema(schema_path, schema_kind=schema_kind)
    write_pandas_compatible_data(data_path, data_kind)
    proc = run_validate(
        schema_path,
        data_path,
        backend=backend_flag,
    )
    assert_validate_ok(proc)


# --- validate: polars ---


def _write_polars_schema(path: Path, schema_kind: str) -> None:
    import pandera.polars as pap
    from pandera.io import polars_io

    schema = pap.DataFrameSchema(
        {
            "x": pap.Column(int),
            "y": pap.Column(str),
        }
    )
    if schema_kind == "yaml":
        polars_io.to_yaml(schema, path)
    elif schema_kind == "json":
        polars_io.to_json(schema, path)
    else:
        raise ValueError(schema_kind)


def _write_polars_data(path: Path, data_kind: str) -> None:
    import polars as pl

    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    if data_kind == "csv":
        df.write_csv(path)
    elif data_kind == "parquet":
        df.write_parquet(path)
    elif data_kind == "json":
        df.write_ndjson(path)
    elif data_kind == "feather":
        df.write_ipc(path)
    else:
        raise ValueError(data_kind)


@pytest.mark.parametrize("schema_kind", ["yaml", "json"])
@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json", "feather"])
def test_validate_polars_all_schema_and_data_formats(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
) -> None:
    pytest.importorskip("polars")
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    _write_polars_schema(schema_path, schema_kind)
    _write_polars_data(data_path, data_kind)
    proc = run_validate(schema_path, data_path, backend="polars")
    assert_validate_ok(proc)


# --- validate: modin / dask / pyspark.pandas (pandas-API library tag) ---


@pytest.mark.parametrize("schema_kind", ["yaml", "json"])
@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json", "feather"])
def test_validate_modin_library_matrix(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
) -> None:
    pytest.importorskip("modin")
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    write_pandas_api_schema(
        schema_path,
        schema_kind=schema_kind,
        dataframe_library="modin",
    )
    write_pandas_compatible_data(data_path, data_kind)
    proc = run_validate(schema_path, data_path, backend="modin")
    assert_validate_ok(proc)


@pytest.mark.parametrize("schema_kind", ["yaml", "json"])
@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json"])
def test_validate_dask_library_matrix(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
) -> None:
    pytest.importorskip("dask")
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    write_pandas_api_schema(
        schema_path,
        schema_kind=schema_kind,
        dataframe_library="dask",
    )
    write_pandas_compatible_data(
        data_path,
        data_kind,
        dask_json_lines=(data_kind == "json"),
    )
    proc = run_validate(schema_path, data_path, backend="dask")
    assert_validate_ok(proc)


@pytest.mark.skipif(
    _SKIP_SPARK_CLI_COMBOS,
    reason=(
        "Set PANDERA_RUN_SPARK_CLI=1 to run Spark CLI combos (requires working "
        "local Spark/JVM)."
    ),
)
@pytest.mark.parametrize("schema_kind", ["yaml", "json"])
@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json"])
def test_validate_pyspark_pandas_library_matrix(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
) -> None:
    pytest.importorskip("pyspark")
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    # Spark's csv reader infers int32 for small integers while the schema's
    # int dtype serializes as int64 — coerce so the dtype check passes
    # across Spark versions (CLI-inferred schemas also set coerce=True).
    write_pandas_api_schema(
        schema_path,
        schema_kind=schema_kind,
        dataframe_library="pyspark.pandas",
        coerce=True,
    )
    write_pandas_compatible_data(data_path, data_kind)
    proc = run_validate(
        schema_path,
        data_path,
        backend="pyspark.pandas",
    )
    assert_validate_ok(proc)


# --- validate: ibis ---


@pytest.mark.parametrize("schema_kind", ["yaml", "json"])
@pytest.mark.parametrize("data_kind", ["csv", "parquet"])
def test_validate_ibis_schema_and_data_formats(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
) -> None:
    pytest.importorskip("ibis")
    import ibis.expr.datatypes as dt

    import pandera.ibis as pa_ib
    from pandera.io import ibis_io

    schema = pa_ib.DataFrameSchema(
        {
            "x": pa_ib.Column(dt.int64),
            "y": pa_ib.Column(dt.String),
        }
    )
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    if schema_kind == "yaml":
        ibis_io.to_yaml(schema, schema_path)
    else:
        ibis_io.to_json(schema, schema_path)
    write_pandas_compatible_data(data_path, data_kind)
    proc = run_validate(schema_path, data_path, backend="ibis")
    assert_validate_ok(proc)


# --- validate: pyspark.sql (subset: JVM cost per subprocess) ---


@pytest.mark.skipif(
    _SKIP_SPARK_CLI_COMBOS,
    reason=(
        "Set PANDERA_RUN_SPARK_CLI=1 to run Spark CLI combos (requires working "
        "local Spark/JVM)."
    ),
)
@pytest.mark.parametrize(
    ("schema_kind", "data_kind"),
    [
        ("yaml", "csv"),
        ("json", "parquet"),
        ("yaml", "json"),
    ],
)
def test_validate_pyspark_sql_representative_combos(
    tmp_path: Path,
    schema_kind: str,
    data_kind: str,
) -> None:
    pytest.importorskip("pyspark")
    import pandera.pyspark as pa_sp
    from pandera.io import pyspark_sql_io

    schema = pa_sp.DataFrameSchema(
        {
            "x": pa_sp.Column(int),
            "y": pa_sp.Column(str),
        }
    )
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    data_path = tmp_path / f"data.{data_kind}"
    if schema_kind == "yaml":
        pyspark_sql_io.to_yaml(schema, schema_path)
    else:
        pyspark_sql_io.to_json(schema, schema_path)
    write_pandas_compatible_data(
        data_path,
        data_kind,
        dask_json_lines=(data_kind == "json"),
    )
    proc = run_validate(schema_path, data_path, backend="pyspark.sql")
    assert_validate_ok(proc)


# --- infer: pandas backend ---


@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json", "feather"])
@pytest.mark.parametrize("out_kind", ["yaml", "json"])
def test_infer_pandas_data_and_schema_output_formats(
    tmp_path: Path,
    data_kind: str,
    out_kind: str,
) -> None:
    data_path = tmp_path / f"data.{data_kind}"
    out_ext = ".yaml" if out_kind == "yaml" else ".json"
    out_path = tmp_path / f"inferred{out_ext}"
    write_pandas_compatible_data(data_path, data_kind)
    proc = run_infer(data_path, out_path, backend="pandas")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_path.is_file()
    raw = out_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) if out_kind == "yaml" else json.loads(raw)
    assert payload["schema_type"] == "dataframe"
    assert payload["api"] == "pandas"


# --- infer: polars backend ---


@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json", "feather"])
@pytest.mark.parametrize("out_kind", ["yaml", "json"])
def test_infer_polars_data_and_schema_output_formats(
    tmp_path: Path,
    data_kind: str,
    out_kind: str,
) -> None:
    pytest.importorskip("polars")
    data_path = tmp_path / f"data.{data_kind}"
    out_ext = ".yaml" if out_kind == "yaml" else ".json"
    out_path = tmp_path / f"inferred{out_ext}"
    _write_polars_data(data_path, data_kind)
    proc = run_infer(data_path, out_path, backend="polars")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_path.is_file()
    raw = out_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) if out_kind == "yaml" else json.loads(raw)
    assert payload["schema_type"] == "polars_dataframe"
    assert payload["api"] == "polars"


# --- infer: modin / dask (pandas-API loaders) ---


@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json", "feather"])
@pytest.mark.parametrize("out_kind", ["yaml", "json"])
def test_infer_modin_backend_data_and_schema_output_formats(
    tmp_path: Path,
    data_kind: str,
    out_kind: str,
) -> None:
    pytest.importorskip("modin")
    data_path = tmp_path / f"data.{data_kind}"
    out_ext = ".yaml" if out_kind == "yaml" else ".json"
    out_path = tmp_path / f"inferred{out_ext}"
    write_pandas_compatible_data(data_path, data_kind)
    proc = run_infer(data_path, out_path, backend="modin")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    raw = out_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) if out_kind == "yaml" else json.loads(raw)
    assert payload["schema_type"] == "dataframe"
    assert payload.get("dataframe_library") == "modin"
    assert payload["api"] == "modin"


@pytest.mark.parametrize("data_kind", ["csv", "parquet", "json"])
@pytest.mark.parametrize("out_kind", ["yaml", "json"])
def test_infer_dask_backend_data_and_schema_output_formats(
    tmp_path: Path,
    data_kind: str,
    out_kind: str,
) -> None:
    pytest.importorskip("dask")
    data_path = tmp_path / f"data.{data_kind}"
    out_ext = ".yaml" if out_kind == "yaml" else ".json"
    out_path = tmp_path / f"inferred{out_ext}"
    write_pandas_compatible_data(
        data_path,
        data_kind,
        dask_json_lines=(data_kind == "json"),
    )
    proc = run_infer(data_path, out_path, backend="dask")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    raw = out_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) if out_kind == "yaml" else json.loads(raw)
    assert payload["schema_type"] == "dataframe"
    assert payload.get("dataframe_library") == "dask"
    assert payload["api"] == "dask"


# --- infer: schema file format override (.txt + --format) ---


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("out_fmt", ["yaml", "json"])
def test_infer_format_flag_with_plain_output_path(
    tmp_path: Path,
    backend: str,
    out_fmt: str,
) -> None:
    if backend == "polars":
        pytest.importorskip("polars")
    data_path = tmp_path / "data.csv"
    write_pandas_compatible_data(data_path, "csv")
    out_path = tmp_path / "schema_out"
    proc = run_infer(
        data_path,
        out_path,
        backend=backend,
        output_format=out_fmt,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_path.is_file()
    raw = out_path.read_text(encoding="utf-8")
    expected_type = "polars_dataframe" if backend == "polars" else "dataframe"
    expected_api = "polars" if backend == "polars" else "pandas"
    payload = yaml.safe_load(raw) if out_fmt == "yaml" else json.loads(raw)
    assert payload["schema_type"] == expected_type
    assert payload["api"] == expected_api


# --- generate: pandas schema yaml/json × csv output ---


@pytest.mark.parametrize("schema_kind", ["yaml", "json"])
def test_generate_pandas_schema_file_formats_to_csv(
    tmp_path: Path,
    schema_kind: str,
) -> None:
    pytest.importorskip("hypothesis")
    import subprocess
    import sys

    import pandera.pandas as pa
    from pandera.io import pandas_io

    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    ext = ".yaml" if schema_kind == "yaml" else ".json"
    schema_path = tmp_path / f"schema{ext}"
    out_path = tmp_path / "out.csv"
    if schema_kind == "yaml":
        pandas_io.to_yaml(schema, schema_path)
    else:
        pandas_io.to_json(schema, schema_path)
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
    import pandas as pd

    df = pd.read_csv(out_path)
    assert len(df) == 3
    assert "a" in df.columns
