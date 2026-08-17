"""``pandera generate`` — synthetic data from a serialized schema."""

from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from . import rich_report
from .common import load_raw_schema

__all__ = ["GenerateDataFormat", "generate"]


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


#: CSV is a text format and cannot carry NUL. Worse, on Python < 3.12 the
#: csv writer stores an unset ``escapechar`` as 0, so a NUL in the data
#: compares equal to it and the write fails outright with "need to escape,
#: but no escapechar set" regardless of the quoting mode. Substitute the
#: Unicode replacement character, which keeps string lengths intact so
#: inferred ``str_length`` checks still hold for the generated sample.
_CSV_UNREPRESENTABLE = {"\x00": "�"}


def _sanitize_for_csv(df: Any) -> Any:
    """Replace characters CSV cannot represent; returns a copy if changed."""

    def _clean(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for bad, good in _CSV_UNREPRESENTABLE.items():
            if bad in value:
                value = value.replace(bad, good)
        return value

    import pandas as pd

    def _may_hold_strings(series: Any) -> bool:
        # Spelled with the pandas type-check API rather than a dtype
        # allowlist: the default string dtype differs across pandas versions
        # (``object`` on 2.x, ``str`` on 3.x), and missing it here silently
        # skips sanitizing and writes a raw NUL.
        return pd.api.types.is_object_dtype(
            series
        ) or pd.api.types.is_string_dtype(series)

    out = df
    for col in df.columns:
        if _may_hold_strings(df[col]):
            cleaned = df[col].map(_clean)
            if not cleaned.equals(df[col]):
                if out is df:
                    out = df.copy()
                out[col] = cleaned

    renamed = {c: _clean(c) for c in out.columns}
    if any(k != v for k, v in renamed.items()):
        out = out.rename(columns=renamed)
    return out


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
        df = _sanitize_for_csv(df)
    if writer_key == "csv":
        # Quote every field: under QUOTE_MINIMAL a bare "\r" is written raw,
        # and readers then treat it as a row break, silently splitting one
        # record into two.
        df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_ALL,
        )
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

    raw = load_raw_schema(schema_path)
    try:
        schema_kind, schema_obj = _deserialize_schema_generate(raw)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.secho(f"Could not load schema:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    writer_key = _resolve_generate_writer(
        output, output_format, schema_kind=schema_kind
    )

    try:
        if schema_kind == "pandas":
            data_obj = _generate_pandas_example(schema_obj, size)
        else:
            data_obj = _generate_xarray_example(schema_obj, size)
    except typer.Exit:
        raise
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
    except (OSError, ImportError, ValueError) as exc:
        typer.secho(f"Could not write {output}:\n{exc}", err=True)
        raise typer.Exit(1) from exc

    rich_report.print_generate_summary(
        schema_path=schema_path,
        schema_kind=schema_kind,
        schema_obj=schema_obj,
        size=size,
        output=output,
        writer_key=writer_key,
        data_obj=data_obj,
    )
