"""Command-line interface implementation (requires ``typer``; see ``pandera[cli]``)."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    help="Pandera command-line tools.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Pandera command-line tools."""


class BackendName(str, Enum):
    """Supported dataframe libraries for CLI validation."""

    pandas = "pandas"
    modin = "modin"
    dask = "dask"
    pyspark_pandas = "pyspark.pandas"
    pyspark_sql = "pyspark.sql"
    polars = "polars"
    ibis = "ibis"


def _load_raw_schema(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            typer.secho(
                "Reading YAML schemas requires PyYAML. Install with:\n"
                "  pip install 'pandera[io]'",
                err=True,
            )
            raise typer.Exit(1) from exc
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    else:
        typer.secho(
            f"Unsupported schema extension {suffix!r}. "
            "Use .yaml, .yml, or .json.",
            err=True,
        )
        raise typer.Exit(1)
    if not isinstance(raw, dict):
        typer.secho(
            "Schema file must contain a JSON object at the top level.",
            err=True,
        )
        raise typer.Exit(1)
    return raw


def _infer_backend_from_schema(data: dict[str, Any]) -> str:
    st = data.get("schema_type")
    if st in (None, "dataframe"):
        lib = data.get("dataframe_library", "pandas")
        if lib in (None, "pandas"):
            return "pandas"
        if lib == "modin":
            return "modin"
        if lib == "dask":
            return "dask"
        if lib == "pyspark.pandas":
            return "pyspark.pandas"
        typer.secho(
            f"Unsupported dataframe_library {lib!r} for schema_type "
            "'dataframe'. Expected one of: missing/'pandas', 'modin', "
            "'dask', 'pyspark.pandas'.",
            err=True,
        )
        raise typer.Exit(1)
    if st == "polars_dataframe":
        return "polars"
    if st == "pyspark_sql_dataframe":
        return "pyspark.sql"
    if st == "ibis_table":
        return "ibis"
    typer.secho(
        f"Unsupported schema_type {st!r}. "
        "Expected one of: missing/'dataframe' (pandas API), "
        "'polars_dataframe', 'pyspark_sql_dataframe', 'ibis_table'.",
        err=True,
    )
    raise typer.Exit(1)


def _deserialize_schema(data: dict[str, Any]):
    st = data.get("schema_type")
    if st in (None, "dataframe"):
        from pandera.io.pandas_io import deserialize_schema

        return deserialize_schema(data)
    if st == "polars_dataframe":
        from pandera.io.polars_io import deserialize_schema

        return deserialize_schema(data)
    if st == "pyspark_sql_dataframe":
        from pandera.io.pyspark_sql_io import deserialize_schema

        return deserialize_schema(data)
    if st == "ibis_table":
        from pandera.io.ibis_io import deserialize_schema

        return deserialize_schema(data)
    typer.secho(f"Unsupported schema_type {st!r}.", err=True)
    raise typer.Exit(1)


def _load_dataset(path: Path, backend: str):
    """Load on-disk data in a backend-native container."""
    suffix = path.suffix.lower()

    if backend == "pandas":
        import pandas as pd

        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix in (".feather", ".ipc"):
            return pd.read_feather(path)
        typer.secho(
            f"Unsupported data extension {suffix!r} for pandas. "
            "Try .csv, .parquet, .json, or .feather.",
            err=True,
        )
        raise typer.Exit(1)

    if backend == "modin":
        import modin.pandas as pd

        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix in (".feather", ".ipc"):
            return pd.read_feather(path)
        typer.secho(
            f"Unsupported data extension {suffix!r} for modin. "
            "Try .csv, .parquet, .json, or .feather.",
            err=True,
        )
        raise typer.Exit(1)

    if backend == "dask":
        import dask.dataframe as dd

        if suffix == ".csv":
            return dd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return dd.read_parquet(path)
        if suffix == ".json":
            return dd.read_json(path)
        typer.secho(
            f"Unsupported data extension {suffix!r} for dask. "
            "Try .csv, .parquet, or .json.",
            err=True,
        )
        raise typer.Exit(1)

    if backend == "pyspark.pandas":
        import pyspark.pandas as ps

        if suffix == ".csv":
            return ps.read_csv(str(path))
        if suffix in (".parquet", ".pq"):
            return ps.read_parquet(str(path))
        if suffix == ".json":
            return ps.read_json(str(path))
        typer.secho(
            f"Unsupported data extension {suffix!r} for pyspark.pandas. "
            "Try .csv, .parquet, or .json.",
            err=True,
        )
        raise typer.Exit(1)

    if backend == "polars":
        import polars as pl

        if suffix == ".csv":
            return pl.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pl.read_parquet(path)
        if suffix == ".json":
            return pl.read_ndjson(path)
        typer.secho(
            f"Unsupported data extension {suffix!r} for polars. "
            "Try .csv, .parquet, or .json (newline-delimited).",
            err=True,
        )
        raise typer.Exit(1)

    if backend == "pyspark.sql":
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("pandera-cli").getOrCreate()
        if suffix == ".csv":
            return (
                spark.read.option("header", True)
                .option("inferSchema", True)
                .csv(str(path))
            )
        if suffix in (".parquet", ".pq"):
            return spark.read.parquet(str(path))
        if suffix == ".json":
            return spark.read.json(str(path))
        typer.secho(
            f"Unsupported data extension {suffix!r} for pyspark.sql. "
            "Try .csv, .parquet, or .json.",
            err=True,
        )
        raise typer.Exit(1)

    if backend == "ibis":
        import ibis

        if suffix == ".csv":
            return ibis.read_csv(str(path))
        if suffix in (".parquet", ".pq"):
            return ibis.read_parquet(str(path))
        typer.secho(
            f"Unsupported data extension {suffix!r} for ibis. "
            "Try .csv or .parquet (requires a suitable Ibis backend).",
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(f"Unknown backend {backend!r}.", err=True)
    raise typer.Exit(1)


@app.command()
def validate(
    schema: Path = typer.Option(
        ...,
        "--schema",
        "-s",
        help="Path to schema file (.yaml, .yml, or .json).",
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to the dataset (format must match backend).",
    ),
    backend: BackendName | None = typer.Option(
        None,
        "--backend",
        "-b",
        help=(
            "Dataframe library to use. Default: inferred from schema_type "
            "(and dataframe_library for pandas-API schemas) in the schema "
            "file."
        ),
    ),
) -> None:
    """Validate a file against a serialized schema (YAML/JSON)."""
    if not schema.is_file():
        typer.secho(f"Schema file not found: {schema}", err=True)
        raise typer.Exit(1)
    if not data.is_file():
        typer.secho(f"Data file not found: {data}", err=True)
        raise typer.Exit(1)

    raw = _load_raw_schema(schema)
    inferred = _infer_backend_from_schema(raw)
    if backend is not None and backend.value != inferred:
        typer.secho(
            f"--backend {backend.value!r} does not match schema "
            f"(schema_type implies {inferred!r}). Omit --backend to use "
            "the schema file, or fix the mismatch.",
            err=True,
        )
        raise typer.Exit(1)
    chosen = backend.value if backend is not None else inferred

    schema_obj = _deserialize_schema(raw)
    obj = _load_dataset(data, chosen)

    try:
        schema_obj.validate(obj)
    except Exception as exc:
        typer.secho(f"Validation failed:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("Validation succeeded.")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
