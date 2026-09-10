"""CLI tests for the Narwhals-powered validation backend (``--backend narwhals``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_validate_polars_backend_narwhals_ok(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_polars_schema(schema_path)
    _write_polars_data(data_path)
    proc = run_validate(schema_path, data_path, backend="narwhals")
    assert_validate_ok(proc, backend="narwhals")


def test_validate_polars_backend_narwhals_failure_exits_nonzero(
    tmp_path: Path,
) -> None:
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_polars_schema(schema_path)
    _write_polars_data(data_path, valid=False)
    proc = run_validate(schema_path, data_path, backend="narwhals")
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
    assert_validate_ok(proc, backend="narwhals")


def test_validate_ibis_backend_narwhals_ok(tmp_path: Path) -> None:
    pytest.importorskip("ibis")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_ibis_schema(schema_path)
    import pandas as pd

    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_csv(data_path, index=False)
    proc = run_validate(schema_path, data_path, backend="narwhals")
    assert_validate_ok(proc, backend="narwhals")


def _write_pandas_checked_schema(path: Path) -> None:
    import pandera.pandas as pa
    from pandera.io import pandas_io

    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(int, checks=pa.Check.ge(1)),
            "y": pa.Column(str),
        }
    )
    pandas_io.to_yaml(schema, path)


def _write_pandas_data(path: Path, *, valid: bool = True) -> None:
    import pandas as pd

    if valid:
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    else:
        df = pd.DataFrame({"x": [1, 0], "y": ["a", "b"]})
    df.to_csv(path, index=False)


def test_validate_pandas_backend_narwhals_ok(tmp_path: Path) -> None:
    """The Narwhals backend swaps ``pd.DataFrame`` dispatch, so pandas works.

    Asserting the report names the ``narwhals`` backend proves the
    Narwhals-powered backend actually ran — a silent fallback to the native
    pandas backend would report ``Backend: pandas`` instead.
    """
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    write_pandas_api_schema(schema_path, schema_kind="yaml")
    write_pandas_compatible_data(data_path, "csv")
    proc = run_validate(schema_path, data_path, backend="narwhals")
    assert_validate_ok(proc, backend="narwhals")


def test_validate_pandas_backend_narwhals_failure_exits_nonzero(
    tmp_path: Path,
) -> None:
    """A failing pandas check exits non-zero with a narwhals failure report."""
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_pandas_checked_schema(schema_path)
    _write_pandas_data(data_path, valid=False)
    proc = run_validate(schema_path, data_path, backend="narwhals")
    assert proc.returncode != 0
    assert "Validation failed" in proc.stderr + proc.stdout
    assert "Backend" in proc.stdout, proc.stdout
    assert "narwhals" in proc.stdout, proc.stdout


def test_validate_pandas_narwhals_env_var_ok(tmp_path: Path) -> None:
    """``PANDERA_USE_NARWHALS_BACKEND=True`` routes pandas through Narwhals."""
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    write_pandas_api_schema(schema_path, schema_kind="yaml")
    write_pandas_compatible_data(data_path, "csv")
    proc = run_validate(
        schema_path,
        data_path,
        env={"PANDERA_USE_NARWHALS_BACKEND": "True"},
    )
    assert_validate_ok(proc, backend="narwhals")


def test_validate_pandas_default_uses_native_backend(tmp_path: Path) -> None:
    """Without the narwhals flag, pandas keeps its native backend."""
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    write_pandas_api_schema(schema_path, schema_kind="yaml")
    write_pandas_compatible_data(data_path, "csv")
    proc = run_validate(schema_path, data_path)
    assert_validate_ok(proc, backend="pandas")


@pytest.mark.parametrize("library", ["modin", "dask", "pyspark.pandas"])
def test_validate_narwhals_rejects_other_pandas_like_apis(
    tmp_path: Path, library: str
) -> None:
    """Only ``pd.DataFrame`` is routed through Narwhals, not its cousins."""
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    write_pandas_api_schema(
        schema_path, schema_kind="yaml", dataframe_library=library
    )
    write_pandas_compatible_data(data_path, "csv")
    proc = run_validate(schema_path, data_path, backend="narwhals")
    assert proc.returncode == 1
    assert f"--backend narwhals is not supported for api '{library}'" in (
        proc.stderr + proc.stdout
    )


def test_validate_backend_mismatch_with_schema_api(tmp_path: Path) -> None:
    """``--backend ibis`` is rejected for a schema whose api is ``polars``."""
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    _write_polars_schema(schema_path)
    _write_polars_data(data_path)
    proc = run_validate(schema_path, data_path, backend="ibis")
    assert proc.returncode == 1
    assert "does not match the schema's api 'polars'" in (
        proc.stderr + proc.stdout
    )


def test_pandas_api_schema_file_includes_api_field(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.yaml"
    data_path = tmp_path / "data.csv"
    write_pandas_api_schema(schema_path, schema_kind="yaml")
    write_pandas_compatible_data(data_path, "csv")
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert payload["api"] == "pandas"
    proc = run_validate(schema_path, data_path)
    assert_validate_ok(proc)


def test_polars_schema_file_includes_api_field(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    schema_path = tmp_path / "schema.yaml"
    _write_polars_schema(schema_path)
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert payload["api"] == "polars"
    assert payload["schema_type"] == "polars_dataframe"


def test_ibis_schema_file_includes_api_field(tmp_path: Path) -> None:
    pytest.importorskip("ibis")
    schema_path = tmp_path / "schema.yaml"
    _write_ibis_schema(schema_path)
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert payload["api"] == "ibis"
    assert payload["schema_type"] == "ibis_table"
