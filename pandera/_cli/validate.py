"""``pandera validate`` — validate data against a serialized schema."""

from __future__ import annotations

from pathlib import Path

import typer

from pandera.errors import SchemaError, SchemaErrors

from . import rich_report
from .common import (
    BackendName,
    deserialize_schema,
    enable_narwhals_backend,
    import_accessor_modules,
    infer_backend_from_schema,
    load_dataset,
    load_raw_schema,
)

__all__ = ["validate"]


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
            "Validation backend. A dataframe API (pandas, modin, dask, "
            "pyspark.pandas, polars, ibis, pyspark.sql) must match the "
            "schema's api field; narwhals validates pandas, polars, ibis, "
            "and pyspark.sql schemas through the Narwhals-powered backend "
            "(requires pandera[narwhals]). Default: the schema's api."
        ),
    ),
) -> None:
    """Validate a file against a serialized schema (YAML/JSON).

    On success, prints a summary of schema- and data-level checks (Rich
    tables when Rich is installed). On failure, prints which checks passed or
    failed plus failure details, then exits with a non-zero code.

    Examples:

    Validate CSV file with YAML schema with long form option names

    ```bash
    pandera validate --schema schema.yaml --data data.csv
    ```

    Validate Parquet file with YAML schema with short form option names
    ```
    pandera validate -s schema.yml -d data.parquet
    ```

    Validate JSON file with JSON schema with short form option names
    ```
    python -m pandera validate -s schema.json -d records.json
    ```

    Validate CSV file with YAML schema with Polars backend with long form option
    names
    ```
    pandera validate -s schema.yaml -d data.csv --backend polars
    ```

    Validate a pandas or Polars schema through the Narwhals-powered backend
    ```
    pandera validate -s schema.yaml -d data.csv --backend narwhals
    ```

    ``--backend narwhals`` is equivalent to setting
    ``PANDERA_USE_NARWHALS_BACKEND=True``.

    The report shows the validation backend that actually ran (e.g.
    ``Backend: narwhals`` or ``Backend: pandas``).
    """
    if not schema.is_file():
        typer.secho(f"Schema file not found: {schema}", err=True)
        raise typer.Exit(1)
    if not data.is_file():
        typer.secho(f"Data file not found: {data}", err=True)
        raise typer.Exit(1)

    raw = load_raw_schema(schema)
    api = infer_backend_from_schema(raw)
    if (
        backend is not None
        and backend is not BackendName.narwhals
        and backend.value != api
    ):
        typer.secho(
            f"--backend {backend.value!r} does not match the schema's api "
            f"{api!r}. Omit --backend to use the schema file, or fix the "
            "mismatch.",
            err=True,
        )
        raise typer.Exit(1)

    if backend is BackendName.narwhals:
        enable_narwhals_backend(api)

    schema_obj = deserialize_schema(raw)
    obj = load_dataset(data, api)
    import_accessor_modules(api)

    # The backend actually in use (``narwhals`` when the Narwhals-powered
    # backend swapped the dispatch, otherwise the native backend for the
    # schema's API). Show it in the report so ``--backend narwhals`` is
    # verifiable.
    backend_name = rich_report.backend_label(type(schema_obj.get_backend(obj)))

    try:
        schema_obj.validate(obj, lazy=True)
    except SchemaErrors as exc:
        typer.secho("Validation failed.", err=True)
        rich_report.print_validation_failure(
            schema_obj, exc, backend_name=backend_name
        )
        raise typer.Exit(1) from exc
    except SchemaError as exc:
        typer.secho("Validation failed.", err=True)
        rich_report.print_validation_failure(
            schema_obj, exc, backend_name=backend_name
        )
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.secho(f"Validation failed:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    rich_report.print_validation_success(schema_obj, backend_name=backend_name)
