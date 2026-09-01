"""Shared helpers for the Pandera CLI (schema loading and dataset I/O)."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import typer


class BackendName(str, Enum):
    """Supported backends for CLI validation.

    Values other than ``narwhals`` name the underlying dataframe API
    objects a schema can validate (see the ``api`` field on serialized
    schemas). ``narwhals`` selects the Narwhals-powered validation
    backend for narwhals-compatible schemas.
    """

    pandas = "pandas"
    modin = "modin"
    dask = "dask"
    pyspark_pandas = "pyspark.pandas"
    pyspark_sql = "pyspark.sql"
    polars = "polars"
    ibis = "ibis"
    narwhals = "narwhals"


#: Underlying dataframe APIs a serialized schema can declare (``api``
#: field) and the CLI can load data for.
API_VALUES = (
    "pandas",
    "modin",
    "dask",
    "pyspark.pandas",
    "polars",
    "ibis",
    "pyspark.sql",
)

#: APIs that can validate through the Narwhals-powered backend. The backend
#: swaps ``pd.DataFrame`` dispatch only; the other pandas-like frame types
#: (modin, dask, pyspark.pandas) stay on their native backends.
NARWHALS_COMPATIBLE_APIS = ("pandas", "polars", "ibis", "pyspark.sql")

#: ``schema_type`` values from serialized schemas mapped to the ``api``
#: values compatible with each (``api`` is optional; this is used to
#: cross-check the two fields when both are present).
_SCHEMA_TYPE_APIS: dict[str, tuple[str, ...]] = {
    "dataframe": ("pandas", "modin", "dask", "pyspark.pandas"),
    "polars_dataframe": ("polars",),
    "pyspark_sql_dataframe": ("pyspark.sql",),
    "ibis_table": ("ibis",),
}


def enable_narwhals_backend(api: str) -> None:
    """Enable the Narwhals-powered validation backend for this process.

    Exits with an error if ``api`` is not narwhals-compatible or the
    ``narwhals`` package is not installed. Must be called before the schema
    is deserialized so that backend registration picks up the setting.
    """
    if api not in NARWHALS_COMPATIBLE_APIS:
        typer.secho(
            f"--backend narwhals is not supported for api {api!r}. "
            "The Narwhals backend supports: "
            f"{', '.join(NARWHALS_COMPATIBLE_APIS)}.",
            err=True,
        )
        raise typer.Exit(1)
    try:
        import narwhals.stable.v1  # noqa: F401
    except ImportError as exc:
        typer.secho(
            "--backend narwhals requires the 'narwhals' package. "
            "Install with:\n"
            "  pip install 'pandera[narwhals]'",
            err=True,
        )
        raise typer.Exit(1) from exc

    from pandera.config import set_config

    set_config(use_narwhals_backend=True)


def import_accessor_modules(backend: str) -> None:
    """Import the modules that register the ``.pandera`` dataframe accessor.

    Validation backends attach schema metadata through the ``.pandera``
    accessor, which is registered as an import side effect of per-library
    pandera modules. Those imports don't happen on the CLI's deserialization
    path, so trigger them explicitly for the chosen backend.
    """
    if backend in ("pandas", "modin", "dask", "pyspark.pandas"):
        import pandera.pandas  # noqa: F401

        if backend == "modin":
            from pandera.accessors import modin_accessor  # noqa: F401
        elif backend == "dask":
            from pandera.accessors import dask_accessor  # noqa: F401
        elif backend == "pyspark.pandas":
            from pandera.accessors import pyspark_accessor  # noqa: F401
    elif backend == "pyspark.sql":
        import pandera.pyspark  # noqa: F401
    elif backend == "polars":
        import pandera.polars  # noqa: F401
    elif backend == "ibis":
        import pandera.ibis  # noqa: F401


def load_raw_schema(path: Path) -> dict[str, Any]:
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


def infer_backend_from_schema(data: dict[str, Any]) -> str:
    """Infer the dataframe API of a serialized schema file.

    Prefers the explicit ``api`` field. Falls back to the
    ``schema_type``/``dataframe_library`` mapping for schema files written
    before the ``api`` field existed.
    """
    api = data.get("api")
    if api is not None:
        if api not in API_VALUES:
            typer.secho(
                f"Unsupported api {api!r}. Expected one of: "
                f"{', '.join(API_VALUES)}.",
                err=True,
            )
            raise typer.Exit(1)
        st = data.get("schema_type")
        if st in _SCHEMA_TYPE_APIS and api not in _SCHEMA_TYPE_APIS[st]:
            typer.secho(
                f"api {api!r} does not match schema_type {st!r} "
                f"(expected one of: {', '.join(_SCHEMA_TYPE_APIS[st])}).",
                err=True,
            )
            raise typer.Exit(1)
        return api

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


def deserialize_schema(data: dict[str, Any]):
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


def load_dataset(path: Path, backend: str):
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
        if suffix in (".feather", ".ipc"):
            return pl.read_ipc(path)
        typer.secho(
            f"Unsupported data extension {suffix!r} for polars. "
            "Try .csv, .parquet, .json (newline-delimited), or .feather.",
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
