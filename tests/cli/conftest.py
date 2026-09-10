"""Shared helpers for CLI tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import pandera.pandas as pa
from pandera.io import pandas_io as pandas_io_mod


def sample_pandas_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})


def pandas_api_schema(*, coerce: bool = False) -> Any:
    return pa.DataFrameSchema(
        {
            "x": pa.Column(int),
            "y": pa.Column(str),
        },
        coerce=coerce,
    )


def write_pandas_api_schema(
    path: Path,
    *,
    schema_kind: str,
    dataframe_library: str | None = None,
    coerce: bool = False,
) -> None:
    """Write schema using ``.yaml``, ``.yml``, or ``.json`` (``schema_kind``)."""
    schema = pandas_api_schema(coerce=coerce)
    kw: dict[str, Any] = {}
    if dataframe_library is not None:
        kw["dataframe_library"] = dataframe_library
    if schema_kind == "yaml":
        pandas_io_mod.to_yaml(schema, path, **kw)
    elif schema_kind == "yml":
        pandas_io_mod.to_yaml(schema, path, **kw)
    elif schema_kind == "json":
        pandas_io_mod.to_json(schema, path, **kw)
    else:
        raise ValueError(schema_kind)


def write_pandas_compatible_data(
    path: Path,
    data_kind: str,
    *,
    dask_json_lines: bool = False,
) -> None:
    """Write a small table as csv, parquet, json (records), or feather.

    When ``dask_json_lines`` is True and ``data_kind`` is ``json``, writes
    newline-delimited records so ``dask.dataframe.read_json`` can load it.
    """
    df = sample_pandas_df()
    if data_kind == "csv":
        df.to_csv(path, index=False)
    elif data_kind == "parquet":
        df.to_parquet(path, index=False)
    elif data_kind == "json":
        if dask_json_lines:
            df.to_json(path, orient="records", lines=True)
        else:
            df.to_json(path, orient="records")
    elif data_kind == "feather":
        df.to_feather(path)
    else:
        raise ValueError(data_kind)


def run_validate(
    schema_path: Path,
    data_path: Path,
    *,
    backend: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``pandera validate``; ``backend="narwhals"`` passes
    ``--backend narwhals``."""
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
    if backend is not None:
        cmd.extend(["--backend", backend])
    run_env = None
    if env is not None:
        run_env = {**os.environ, **env}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
    )


def run_infer(
    data_path: Path,
    output_path: Path,
    *,
    backend: str = "pandas",
    output_format: str | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pandera",
        "infer",
        "-d",
        str(data_path),
        "-o",
        str(output_path),
    ]
    if backend != "pandas":
        cmd.extend(["--backend", backend])
    if output_format is not None:
        cmd.extend(["--format", output_format])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def assert_validate_ok(
    proc: subprocess.CompletedProcess[str], backend: str | None = None
) -> None:
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Validation succeeded" in proc.stdout
    if backend is not None:
        out = proc.stdout
        # The report names the validation backend that actually ran, so this
        # also proves the backend selection (e.g. the Narwhals-powered backend
        # swapping ``pd.DataFrame`` dispatch) took effect.
        assert "Backend" in out, out
        assert f"{backend}" in out, out
