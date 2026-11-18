# CLI

The Pandera CLI can **validate** on-disk datasets against a serialized schema
(YAML or JSON), **infer** a schema from data and write YAML, JSON, or a Python
module, or **generate** synthetic tabular or xarray data from a pandas or
xarray schema. It is useful in scripts and CI without writing Python glue code.

## Installation

The CLI depends on [Typer](https://typer.tiangolo.com/). Install Pandera with
the `cli` extra:

```bash
pip install 'pandera[cli]'
```

You still need the appropriate dataframe library for the schema you validate
(for example `pandas` for the default pandas-API schema). YAML schemas require
`pandera[io]` (PyYAML). To validate pandas, Polars, Ibis, or PySpark SQL
schemas through the [Narwhals-powered backend](./narwhals_backend.html),
install `pandera[narwhals]` and pass `--backend narwhals` to
`pandera validate`.

Serialized schemas carry an optional top-level `api` field declaring the
underlying dataframe API of the data to validate (`pandas`, `modin`,
`dask`, `pyspark.pandas`, `polars`, `ibis`, or `pyspark.sql`). The
CLI uses it to choose the data loader and, by default, the validation
backend; it defaults to `pandas` when the field is absent.

Refer to the [CLI guide](./cli_guide.html) to learn how to use the CLI.

## Commands

```
Usage: pandera [OPTIONS] COMMAND [ARGS]...

  Pandera command-line tools.

Options:
  --install-completion    Install completion for the current shell.
  --show-completion       Show completion for the current shell.
  --help                  Show this message and exit.

Commands:
  validate    Validate a file against a serialized schema (YAML/JSON).
  infer       Infer a schema from a data file and write YAML, JSON, or Python.
  generate    Generate synthetic data from a serialized schema (hypothesis).
```

### pandera validate

Validate a file against a serialized schema (YAML/JSON). On success, prints a
summary of schema- and data-level checks (Rich tables when Rich is installed);
on failure, prints which checks passed or failed plus failure details, then
exits with a non-zero code.

```
Usage: pandera validate [OPTIONS]

Options:
  -s, --schema <path>  Path to schema file (.yaml, .yml, or .json). [required]
  -d, --data <path>    Path to the dataset (format must match backend).
                       [required]
  -b, --backend <pandas|modin|dask|pyspark.pandas|pyspark.sql|polars|ibis|narwhals>
                       Validation backend. A dataframe API (pandas, modin,
                       dask, pyspark.pandas, polars, ibis, pyspark.sql) must
                       match the schema's api field; narwhals validates pandas,
                       polars, ibis, and pyspark.sql schemas via the
                       Narwhals-powered backend (requires pandera[narwhals]).
                       Default: the schema's api.
      --help           Show this message and exit.
```

`--backend narwhals` is equivalent to setting `PANDERA_USE_NARWHALS_BACKEND=True`.
The report shows the validation backend that actually ran (e.g. `Backend:
narwhals` or `Backend: pandas`).

### pandera infer

Infer a schema from a data file and write YAML, JSON, or Python. Uses the same
loaders as validate for each `--backend`. Pandas-API backends infer from an
in-memory pandas DataFrame (dask and modin may trigger a full compute /
conversion).

```
Usage: pandera infer [OPTIONS]

Options:
  -d, --data <path>                 Path to the dataset (same extensions as
                                    validate per backend). [required]
  -o, --output <path>               Path to write the inferred schema (.yaml,
                                    .json, or .py). [required]
  -b, --backend <pandas|modin|dask|pyspark.pandas|pyspark.sql|polars|ibis|narwhals>
                                    Dataframe API to use for loading data and
                                    for the output schema API (narwhals is a
                                    validation-only backend and is not accepted
                                    here). [default: pandas]
  -f, --format <yaml|json|py>       Output format. Default: from --output
                                    extension (.yaml/.yml, .json, .py).
      --script-type <schema|model>  If --format py: emit DataFrameSchema
                                    ("schema") or DataFrameModel ("model").
                                    [default: schema]
      --help                        Show this message and exit.
```

### pandera generate

Generate synthetic data from a serialized schema (hypothesis). Requires
`pandera[strategies]` (hypothesis). Only pandas dataframe schemas and xarray
`data_array` / `dataset` schemas are supported; other backends may be added
later.

```
Usage: pandera generate [OPTIONS]

Options:
  -s, --schema <path>  Path to a YAML or JSON schema (pandas or xarray).
                       [required]
  -o, --output <path>  Path to write generated data. Extension selects format:
                       .csv, .json, .parquet, .feather (pandas or xarray);
                       .nc for xarray NetCDF. [required]
  -n, --size <int>     Number of rows for pandas tables; default dimension size
                       for xarray synthetic data. [default: 10]
  -f, --format <csv|json|parquet|feather|netcdf>
                       Output format. Default: from --output extension. Use
                       netcdf for xarray NetCDF when the path has no suffix.
      --help           Show this message and exit.
```

## Supported file combinations

Schema files must be YAML (with PyYAML installed) or JSON. Supported data
extensions depend on the backend; common cases include CSV, Parquet, JSON, and
Feather for pandas-like backends. Ibis is limited to CSV and Parquet in the CLI
(see the implementation for details).

For more on serialization formats, see [Schema persistence](./frictionless.html)
and [IO Utilities](./reference/io.html).
