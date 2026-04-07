"""Command-line interface implementation (requires ``typer``; see ``pandera[cli]``)."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import typer

app = typer.Typer(
    help="Pandera command-line tools.",
    rich_markup_mode="markdown",
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

        return infer_pandas_df_schema(pd_df)
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


class GenerateDataFormat(str, Enum):
    """Output data format for ``generate`` (when ``--format`` is used)."""

    csv = "csv"
    json = "json"
    parquet = "parquet"
    feather = "feather"
    netcdf = "netcdf"


def _deserialize_schema_generate(data: dict[str, Any]) -> tuple[str, Any]:
    """Load a schema for synthetic data generation (pandas or xarray only)."""
    st = data.get("schema_type")
    if st in (None, "dataframe"):
        lib = data.get("dataframe_library", "pandas")
        if lib not in (None, "pandas"):
            typer.secho(
                "`generate` currently supports pandas dataframe_library only "
                f"(got {lib!r}). Other libraries may be supported later.",
                err=True,
            )
            raise typer.Exit(1)
        from pandera.io.pandas_io import deserialize_schema

        return "pandas", deserialize_schema(data)
    if st in ("data_array", "dataset"):
        from pandera.io.xarray_io import deserialize_schema

        return "xarray", deserialize_schema(data)
    typer.secho(
        "`generate` supports schema_type 'dataframe' (pandas), "
        "'data_array', or 'dataset' (xarray); "
        f"got {st!r}.",
        err=True,
    )
    raise typer.Exit(1)


def _resolve_generate_writer(
    output: Path,
    fmt: GenerateDataFormat | None,
    *,
    schema_kind: str,
) -> str:
    """Return writer key: csv, json, parquet, feather, or netcdf."""
    if fmt is not None:
        key = fmt.value
        if key == "netcdf" and schema_kind != "xarray":
            typer.secho(
                "NetCDF output (`.nc`) is only available for xarray schemas "
                "(schema_type data_array or dataset).",
                err=True,
            )
            raise typer.Exit(1)
        return key

    suffix = output.suffix.lower()
    tabular = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".feather": "feather",
        ".ipc": "feather",
    }
    if suffix in tabular:
        return tabular[suffix]
    if suffix in (".nc", ".cdf", ".netcdf"):
        if schema_kind != "xarray":
            typer.secho(
                "NetCDF output is only for xarray schemas; use a tabular "
                "extension (.csv, .parquet, …) for pandas schemas.",
                err=True,
            )
            raise typer.Exit(1)
        return "netcdf"
    typer.secho(
        "Could not infer output format from the file name. "
        "Use a .csv, .json, .parquet, .feather, or for xarray also .nc, "
        "or pass --format.",
        err=True,
    )
    raise typer.Exit(1)


def _generate_pandas_example(schema: Any, size: int) -> Any:
    import warnings

    import hypothesis

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=hypothesis.errors.NonInteractiveExampleWarning,
        )
        return schema.example(size=size)


def _generate_xarray_example(schema: Any, size: int) -> Any:
    import warnings

    import hypothesis

    from pandera.api.xarray.container import DataArraySchema, DatasetSchema
    from pandera.strategies import xarray_strategies as xst

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=hypothesis.errors.NonInteractiveExampleWarning,
        )
        if isinstance(schema, DataArraySchema):
            strat = xst.data_array_schema_strategy(schema, size=size)
        elif isinstance(schema, DatasetSchema):
            strat = xst.dataset_schema_strategy(schema, size=size)
        else:
            typer.secho(
                f"Internal error: unsupported xarray schema {type(schema)!r}.",
                err=True,
            )
            raise typer.Exit(1)
        return strat.example()


def _write_generated_tabular_pandas(
    df: Any, path: Path, writer_key: str
) -> None:
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        typer.secho(
            f"Internal error: expected pandas.DataFrame, got {type(df).__name__}.",
            err=True,
        )
        raise typer.Exit(1)
    if writer_key == "csv":
        df.to_csv(path, index=False)
    elif writer_key == "json":
        df.to_json(path, orient="records")
    elif writer_key == "parquet":
        df.to_parquet(path, index=False)
    elif writer_key == "feather":
        df.to_feather(path)
    else:
        typer.secho(
            f"Internal error: bad tabular writer {writer_key!r}.",
            err=True,
        )
        raise typer.Exit(1)


def _write_generated_data(
    obj: Any,
    *,
    schema_kind: str,
    writer_key: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if schema_kind == "pandas":
        if writer_key == "netcdf":
            typer.secho(
                "NetCDF is not supported for pandas schemas.", err=True
            )
            raise typer.Exit(1)
        _write_generated_tabular_pandas(obj, path, writer_key)
        return

    import xarray as xr

    if writer_key == "netcdf":
        obj.to_netcdf(path)
        return

    try:
        if isinstance(obj, xr.DataArray):
            pdf = obj.to_dataframe(name=obj.name or "data")
        else:
            pdf = obj.to_dataframe()
    except Exception as exc:
        typer.secho(
            f"Could not convert the generated xarray object to a table for "
            f"{writer_key!r} output:\n{exc}\n"
            "Try writing NetCDF (`.nc`) instead.",
            err=True,
        )
        raise typer.Exit(1) from exc

    _write_generated_tabular_pandas(pdf, path, writer_key)


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
    """Validate a file against a serialized schema (YAML/JSON).

    On success, prints "Validation succeeded." and exits 0. On failure, prints
    the error and exits with a non-zero code.

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
    """
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


@app.command()
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
            "Dataframe library to use for loading data and for the output "
            "schema API."
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

    resolved_fmt = _resolve_infer_format(output, output_format)
    chosen = backend.value

    try:
        obj = _load_dataset(data, chosen)
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

    typer.echo(f"Wrote inferred schema to {output}")


@app.command()
def generate(
    schema_path: Path = typer.Option(
        ...,
        "--schema",
        "-s",
        help="Path to a YAML or JSON schema (pandas or xarray).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help=(
            "Path to write generated data. Extension selects format: "
            ".csv, .json, .parquet, .feather (pandas or xarray); "
            ".nc for xarray NetCDF."
        ),
    ),
    size: int = typer.Option(
        10,
        "--size",
        "-n",
        help=(
            "Number of rows for pandas tables; default dimension size for "
            "xarray synthetic data."
        ),
    ),
    output_format: GenerateDataFormat | None = typer.Option(
        None,
        "--format",
        "-f",
        help=(
            "Output format. Default: from ``--output`` extension. "
            "Use ``netcdf`` for xarray NetCDF when the path has no suffix."
        ),
    ),
) -> None:
    """Generate synthetic data from a serialized schema (hypothesis).

    Requires ``pandera[strategies]`` (hypothesis). Only **pandas** dataframe
    schemas and **xarray** ``data_array`` / ``dataset`` schemas are supported;
    other backends may be added later.

    Examples:

    Generate synthetic data from a YAML schema and write to a CSV file
    ```bash
    pandera generate -s schema.yaml -o sample.csv
    ```

    Generate synthetic data from a JSON schema and write to a NetCDF file
    ```bash
    pandera generate --schema ds_schema.json --output data.nc --size 5
    ```
    """
    if size < 1:
        typer.secho("--size must be at least 1.", err=True)
        raise typer.Exit(1)

    if not schema_path.is_file():
        typer.secho(f"Schema file not found: {schema_path}", err=True)
        raise typer.Exit(1)

    from pandera.strategies.base_strategies import HAS_HYPOTHESIS

    if not HAS_HYPOTHESIS:
        typer.secho(
            "Generating data requires hypothesis. Install with:\n"
            "  pip install 'pandera[strategies]'",
            err=True,
        )
        raise typer.Exit(1)

    raw = _load_raw_schema(schema_path)
    try:
        schema_kind, schema_obj = _deserialize_schema_generate(raw)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.secho(f"Could not load schema:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        writer_key = _resolve_generate_writer(
            output, output_format, schema_kind=schema_kind
        )
    except typer.Exit:
        raise

    try:
        if schema_kind == "pandas":
            data_obj = _generate_pandas_example(schema_obj, size)
        else:
            data_obj = _generate_xarray_example(schema_obj, size)
    except ImportError as exc:
        typer.secho(f"Data generation failed:\n{exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.secho(f"Data generation failed:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        _write_generated_data(
            data_obj,
            schema_kind=schema_kind,
            writer_key=writer_key,
            path=output,
        )
    except typer.Exit:
        raise
    except OSError as exc:
        typer.secho(f"Could not write {output}:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Wrote generated data to {output}")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
