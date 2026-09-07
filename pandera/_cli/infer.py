"""``pandera infer`` — infer a schema from a data file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import typer

from . import rich_report
from .common import API_VALUES, BackendName, load_dataset

__all__ = ["InferFormat", "ScriptType", "infer"]


class InferFormat(str, Enum):
    """Serialized schema output format for ``infer``."""

    yaml = "yaml"
    json = "json"
    py = "py"


class ScriptType(str, Enum):
    """Python script style for ``infer --format py``."""

    schema = "schema"
    model = "model"


def _script_type_for_io(st: ScriptType) -> Literal["schema", "model"]:
    """Map CLI enum to :func:`to_script` keyword (for static typing)."""
    if st is ScriptType.schema:
        return "schema"
    if st is ScriptType.model:
        return "model"
    raise AssertionError(st)


def _resolve_infer_format(
    output: Path,
    fmt: InferFormat | None,
) -> InferFormat:
    if fmt is not None:
        return fmt
    suffix = output.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return InferFormat.yaml
    if suffix == ".json":
        return InferFormat.json
    if suffix == ".py":
        return InferFormat.py
    typer.secho(
        "Could not infer output format from the file name. "
        "Use --format yaml|json|py or use a .yaml, .yml, .json, or .py path.",
        err=True,
    )
    raise typer.Exit(1)


def _pandas_df_for_inference(obj: Any, backend: str) -> Any:
    import pandas as pd

    if backend == "pandas":
        if not isinstance(obj, pd.DataFrame):
            typer.secho(
                "Internal error: expected pandas.DataFrame for pandas "
                f"backend, got {type(obj).__name__}.",
                err=True,
            )
            raise typer.Exit(1)
        return obj
    if backend == "modin":
        to_pd = getattr(obj, "to_pandas", None) or getattr(
            obj, "_to_pandas", None
        )
        if to_pd is None:
            typer.secho(
                "Cannot convert modin DataFrame to pandas for inference.",
                err=True,
            )
            raise typer.Exit(1)
        return to_pd()
    if backend == "dask":
        return obj.compute()
    if backend == "pyspark.pandas":
        return obj.to_pandas()
    typer.secho(
        f"Internal error: unexpected pandas-API backend {backend!r}.",
        err=True,
    )
    raise typer.Exit(1)


def _infer_dataframe_schema_cli(obj: Any, backend: str) -> Any:
    if backend in ("pandas", "modin", "dask", "pyspark.pandas"):
        pd_df = _pandas_df_for_inference(obj, backend)
        from pandera.schema_inference.pandas import (
            infer_dataframe_schema as infer_pandas_df_schema,
        )

        return infer_pandas_df_schema(pd_df, infer_str_length=True)
    if backend == "polars":
        from pandera.schema_inference.polars import (
            infer_dataframe_schema as infer_polars_df_schema,
        )

        return infer_polars_df_schema(obj)
    if backend == "pyspark.sql":
        from pandera.schema_inference.pyspark import (
            infer_dataframe_schema as infer_pyspark_df_schema,
        )

        return infer_pyspark_df_schema(obj)
    if backend == "ibis":
        from pandera.schema_inference.ibis import (
            infer_dataframe_schema as infer_ibis_df_schema,
        )

        return infer_ibis_df_schema(obj)
    typer.secho(f"Unknown backend {backend!r}.", err=True)
    raise typer.Exit(1)


def _dataframe_library_tag(backend: str) -> str | None:
    if backend == "pandas":
        return None
    if backend in ("modin", "dask", "pyspark.pandas"):
        return backend
    return None


def _write_inferred_schema(
    schema: Any,
    *,
    backend: str,
    output: Path,
    fmt: InferFormat,
    script_type: ScriptType,
    minimal: bool,
) -> None:
    if fmt == InferFormat.yaml:
        if backend in ("pandas", "modin", "dask", "pyspark.pandas"):
            from pandera.io import pandas_io

            pandas_io.to_yaml(
                schema,
                output,
                dataframe_library=_dataframe_library_tag(backend),
                minimal=minimal,
            )
        elif backend == "polars":
            from pandera.io import polars_io

            polars_io.to_yaml(schema, output, minimal=minimal)
        elif backend == "pyspark.sql":
            from pandera.io import pyspark_sql_io

            pyspark_sql_io.to_yaml(schema, output, minimal=minimal)
        elif backend == "ibis":
            from pandera.io import ibis_io

            ibis_io.to_yaml(schema, output, minimal=minimal)
        return

    if fmt == InferFormat.json:
        if backend in ("pandas", "modin", "dask", "pyspark.pandas"):
            from pandera.io import pandas_io

            pandas_io.to_json(
                schema,
                output,
                dataframe_library=_dataframe_library_tag(backend),
                minimal=minimal,
            )
        elif backend == "polars":
            from pandera.io import polars_io

            polars_io.to_json(schema, output, minimal=minimal)
        elif backend == "pyspark.sql":
            from pandera.io import pyspark_sql_io

            pyspark_sql_io.to_json(schema, output, minimal=minimal)
        elif backend == "ibis":
            from pandera.io import ibis_io

            ibis_io.to_json(schema, output, minimal=minimal)
        return

    if fmt == InferFormat.py:
        st_kw = _script_type_for_io(script_type)
        if backend in ("pandas", "modin", "dask", "pyspark.pandas"):
            from pandera.io import pandas_io

            pandas_io.to_script(
                schema,
                output,
                minimal=minimal,
                script_type=st_kw,
            )
        elif backend == "polars":
            from pandera.io import polars_io

            polars_io.to_script(
                schema,
                output,
                minimal=minimal,
                script_type=st_kw,
            )
        elif backend == "pyspark.sql":
            from pandera.io import pyspark_sql_io

            pyspark_sql_io.to_script(
                schema,
                output,
                minimal=minimal,
                script_type=st_kw,
            )
        elif backend == "ibis":
            from pandera.io import ibis_io

            ibis_io.to_script(
                schema,
                output,
                minimal=minimal,
                script_type=st_kw,
            )
        return

    typer.secho(f"Unsupported output format {fmt!r}.", err=True)
    raise typer.Exit(1)


def infer(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to the dataset (same extensions as ``validate`` per backend).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write the inferred schema (.yaml, .json, or .py).",
    ),
    backend: BackendName = typer.Option(
        BackendName.pandas,
        "--backend",
        "-b",
        help=(
            "Dataframe API to use for loading data and for the output "
            "schema API (``narwhals`` is a validation-only backend and is "
            "not accepted here)."
        ),
    ),
    output_format: InferFormat | None = typer.Option(
        None,
        "--format",
        "-f",
        help=(
            "Output format. Default: from ``--output`` extension "
            "(.yaml/.yml, .json, .py)."
        ),
    ),
    script_type: ScriptType = typer.Option(
        ScriptType.schema,
        "--script-type",
        help=(
            'If ``--format py``: emit ``DataFrameSchema`` ("schema") or '
            '``DataFrameModel`` ("model").'
        ),
    ),
) -> None:
    """Infer a schema from a data file and write YAML, JSON, or Python.

    Uses the same loaders as ``validate`` for each ``--backend``. Pandas-API
    backends infer from an in-memory pandas DataFrame (dask and modin may
    trigger a full ``compute`` / conversion).

    Examples:

    Infer a schema from a CSV file and write to a YAML file
    ```bash
    pandera infer -d data.csv -o schema.yaml
    ```

    Infer a schema from a Parquet file and write to a JSON file
    ```bash
    pandera infer --data table.parquet --output schema.json --backend polars
    ```

    Infer a schema from a CSV file and write to a Python model file
    ```bash
    pandera infer -d data.csv -o model.py --format py --script-type model
    ```
    """
    if not data.is_file():
        typer.secho(f"Data file not found: {data}", err=True)
        raise typer.Exit(1)

    if backend is BackendName.narwhals:
        typer.secho(
            "--backend narwhals selects a validation backend and cannot be "
            "used with `infer`. Pass the dataframe API instead, e.g. "
            f"{', '.join(API_VALUES)}.",
            err=True,
        )
        raise typer.Exit(1)

    resolved_fmt = _resolve_infer_format(output, output_format)
    chosen = backend.value

    try:
        obj = load_dataset(data, chosen)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.secho(f"Failed to load data:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        schema_obj = _infer_dataframe_schema_cli(obj, chosen)
    except Exception as exc:
        typer.secho(f"Schema inference failed:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_inferred_schema(
            schema_obj,
            backend=chosen,
            output=output,
            fmt=resolved_fmt,
            script_type=script_type,
            minimal=True,
        )
    except typer.Exit:
        raise
    except ImportError as exc:
        typer.secho(
            f"Could not write output ({exc}). "
            "For YAML, install PyYAML (e.g. pip install 'pandera[io]'). "
            "For .py scripts, install black (pip install black).",
            err=True,
        )
        raise typer.Exit(1) from exc
    except OSError as exc:
        typer.secho(f"Could not write {output}:\n{exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.secho(
            f"Could not write the inferred schema to {output}:\n{exc}",
            err=True,
        )
        raise typer.Exit(1) from exc

    rich_report.print_infer_summary(
        data_path=data,
        backend=chosen,
        obj=obj,
        schema=schema_obj,
        output=output,
        fmt=resolved_fmt.value,
    )
