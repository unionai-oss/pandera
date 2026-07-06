"""CLI tests for the Narwhals-powered validation backend (``--use-narwhals``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cli.conftest import (
    assert_validate_ok,
    run_validate,
    write_pandas_api_schema,
    write_pandas_compatible_data,
)

pytest.importorskip("narwhals")


def _write_polars_schema(path: Path) -> None:
    import pandera.polars as pap
    from pandera.io import polars_io

    schema = pap.DataFrameSchema(
        {
            "x": pap.Column(int),
            "y": pap.Column(str),
        }
    )
    polars_io.to_yaml(schema, path)


def _write_polars_data(path: Path, *, valid: bool = True) -> None:
    import polars as pl

    if valid:
        df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    else:
        df = pl.DataFrame({"x": [1, 2], "y": [1, 2]})
    df.write_csv(path)


def _write_ibis_schema(path: Path) -> None:
    import pandera.ibis as pai
    from pandera.io import ibis_io

    schema = pai.DataFrameSchema(
        {
            "x": pai.Column(int),
            "y": pai.Column(str),
        }
    )
    ibis_io.to_yaml(schema, path)


def test_validate_polars_use_narwhals_ok(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_polars_schema(schema_path)
    _write_polars_data(data_path)
    proc = run_validate(schema_path, data_path, use_narwhals=True)
    assert_validate_ok(proc)


def test_validate_polars_use_narwhals_failure_exits_nonzero(
    tmp_path: Path,
) -> None:
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_polars_schema(schema_path)
    _write_polars_data(data_path, valid=False)
    proc = run_validate(schema_path, data_path, use_narwhals=True)
    assert proc.returncode != 0
    assert "Validation failed" in proc.stderr + proc.stdout


def test_validate_polars_narwhals_env_var_ok(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_polars_schema(schema_path)
    _write_polars_data(data_path)
    proc = run_validate(
        schema_path,
        data_path,
        env={"PANDERA_USE_NARWHALS_BACKEND": "True"},
    )
    assert_validate_ok(proc)


def test_validate_ibis_use_narwhals_ok(tmp_path: Path) -> None:
    pytest.importorskip("ibis")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_ibis_schema(schema_path)
    import pandas as pd

    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_csv(data_path, index=False)
    proc = run_validate(schema_path, data_path, use_narwhals=True)
    assert_validate_ok(proc)


def test_validate_use_narwhals_rejects_pandas_backend(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    write_pandas_api_schema(schema_path, schema_kind="yaml")
    write_pandas_compatible_data(data_path, "csv")
    proc = run_validate(schema_path, data_path, use_narwhals=True)
    assert proc.returncode == 1
    assert "--use-narwhals is not supported for backend 'pandas'" in (
        proc.stderr + proc.stdout
    )
