(cli-guide)=

# Validating data with the CLI

This page is a **copy-paste** tour of the `pandera` command-line interface using
**pandas**, **CSV** data, and a **JSON** schema (no PyYAML needed). Run the
commands in order in a terminal.

## Install Pandera with the CLI and strategies extras

Install pandas, the CLI (Typer and Rich), and **Hypothesis** (needed for
`pandera generate` later in this guide):

```bash
pip install 'pandera[pandas,cli,strategies]'
```

## Create a sample dataset

Write a small CSV to `/tmp/dataset.csv`:

```bash
cat <<'EOF' > /tmp/dataset.csv
id,name
1,alice
2,bob
3,carol
EOF
```

## Infer a schema from the data

Write an inferred schema to `/tmp/schema.json`:

```bash
pandera infer -d /tmp/dataset.csv -o /tmp/schema.yaml
```

You can view the yaml schema with

```bash
cat /tmp/schema.yaml
```

```yaml
schema_type: dataframe
columns:
  id:
    dtype: int64
    greater_than_or_equal_to: 1.0
    less_than_or_equal_to: 3.0
  name:
    dtype: object
    str_length:
      min_value: 3
      max_value: 5
      exact_value: null
index:
- dtype: int64
  greater_than_or_equal_to: 0.0
  less_than_or_equal_to: 2.0
coerce: true
```

You should see a short summary (paths, shape, columns) and a line that the
schema was written successfully.

## Validate the good data

```bash
pandera validate -s /tmp/schema.yaml -d /tmp/dataset.csv
```

```
╭──────────────────────────── Validation succeeded ────────────────────────────╮
│                                    Checks                                    │
│ ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓ │
│ ┃ Level   ┃ Target    ┃ Requirement                              ┃ Status  ┃ │
│ ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩ │
│ │ data    │ index     │ dtype: int64                             │ passed  │ │
│ │ data    │ index     │ non-null values                          │ passed  │ │
│ │ data    │ id        │ column present                           │ passed  │ │
│ │ data    │ id        │ dtype: int64                             │ passed  │ │
│ │ data    │ id        │ non-null values                          │ passed  │ │
│ │ data    │ id        │ check: greater_than_or_equal_to          │ passed  │ │
│ │ data    │ id        │ check: less_than_or_equal_to             │ passed  │ │
│ │ data    │ name      │ column present                           │ passed  │ │
│ │ data    │ name      │ dtype: object                            │ passed  │ │
│ │ data    │ name      │ non-null values                          │ passed  │ │
│ │ data    │ name      │ check: str_length                        │ passed  │ │
│ └─────────┴───────────┴──────────────────────────────────────────┴─────────┘ │
╰─────────── All listed schema- and data-level requirements passed. ───────────╯
```

You should see a **Validation succeeded** report (Rich tables if Rich is
installed) and exit code `0`.

## Create data that fails validation

The inferred schema expects `id` to be numeric and `name` to be text. The
following file puts a non-numeric value in `id`:

```bash
cat <<'EOF' > /tmp/invalid_dataset.csv
id,name
x,bob
2,carol
EOF
```

## Validate the invalid data

```bash
pandera validate -s /tmp/schema.yaml -d /tmp/invalid_dataset.csv
```


```
╭───────────────────────────── Validation failed ──────────────────────────────╮
│                                Check results                                 │
│ ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓ │
│ ┃ Level   ┃ Target    ┃ Requirement                              ┃ Status  ┃ │
│ ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩ │
│ │ data    │ index     │ dtype: int64                             │ passed  │ │
│ │ data    │ index     │ non-null values                          │ passed  │ │
│ │ data    │ id        │ column present                           │ passed  │ │
│ │ data    │ id        │ dtype: int64                             │ failed  │ │
│ │ data    │ id        │ non-null values                          │ passed  │ │
│ │ data    │ id        │ check: greater_than_or_equal_to          │ failed  │ │
│ │ data    │ id        │ check: less_than_or_equal_to             │ failed  │ │
│ │ data    │ name      │ column present                           │ passed  │ │
│ │ data    │ name      │ dtype: object                            │ passed  │ │
│ │ data    │ name      │ non-null values                          │ passed  │ │
│ │ data    │ name      │ check: str_length                        │ passed  │ │
│ └─────────┴───────────┴──────────────────────────────────────────┴─────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────╮
│                               Failure details                                │
│ ┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓ │
│ ┃ Level ┃ Target ┃ Reason           ┃ Failure cases    ┃ Message           ┃ │
│ ┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩ │
│ │ data  │ —      │ dtype_coercion_… │ index            │ Error while       │ │
│ │       │        │                  │ failure_case 0 0 │ coercing 'id' to  │ │
│ │       │        │                  │ x                │ type int64: Could │ │
│ │       │        │                  │                  │ not coerce <class │ │
│ │       │        │                  │                  │ 'pandas.core.seri │ │
│ │       │        │                  │                  │ es.Series'>       │ │
│ │       │        │                  │                  │ data_…            │ │
│ │ data  │ id     │ wrong_dtype      │ index            │ expected series   │ │
│ │       │        │                  │ failure_case 0 0 │ 'id' to have type │ │
│ │       │        │                  │ x                │ int64, got object │ │
│ │ data  │ id     │ check_error      │ TypeError("'>='  │ Error while       │ │
│ │       │        │                  │ not supported    │ executing check   │ │
│ │       │        │                  │ between          │ function:         │ │
│ │       │        │                  │ instances of     │ TypeError("'>='   │ │
│ │       │        │                  │ 'str' and        │ not supported     │ │
│ │       │        │                  │ 'float'")        │ between instances │ │
│ │       │        │                  │                  │ of 'str' and …    │ │
│ │ data  │ id     │ check_error      │ TypeError("'<='  │ Error while       │ │
│ │       │        │                  │ not supported    │ executing check   │ │
│ │       │        │                  │ between          │ function:         │ │
│ │       │        │                  │ instances of     │ TypeError("'<='   │ │
│ │       │        │                  │ 'str' and        │ not supported     │ │
│ │       │        │                  │ 'float'")        │ between instances │ │
│ │       │        │                  │                  │ of 'str' and …    │ │
│ └───────┴────────┴──────────────────┴──────────────────┴───────────────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯
```

This should exit with a **non-zero** status. You should see **Validation failed**
on standard error plus tables listing which checks passed or failed and failure
details (exact layout depends on your Pandera and Rich versions).

## Validate through the Narwhals backend

Polars, Ibis, and PySpark SQL schemas can validate through the
{ref}`Narwhals-powered backend <narwhals-backend>` by passing `--use-narwhals`
to `pandera validate`. Install the `narwhals` extra alongside the dataframe
library you use:

```bash
pip install 'pandera[cli,narwhals,polars]'
```

Then pass the flag when validating a Polars, Ibis, or PySpark SQL schema:

```bash
pandera validate -s /tmp/polars_schema.yaml -d /tmp/dataset.csv --use-narwhals
```

This is equivalent to running the CLI with the
`PANDERA_USE_NARWHALS_BACKEND=True` environment variable set:

```bash
PANDERA_USE_NARWHALS_BACKEND=True pandera validate -s /tmp/polars_schema.yaml -d /tmp/dataset.csv
```

Passing `--use-narwhals` with a pandas-API schema (pandas, modin, dask, or
pyspark.pandas) exits with an error, since the Narwhals backend only powers
the Polars, Ibis, and PySpark SQL integrations.

:::{important}
**Other backends and formats**

The same CLI supports **Polars**, **PySpark**, **Ibis**, and other loaders
depending on how you install Pandera and which `--backend` you pass to
`infer` / `validate`. Schema files can be **YAML** (install `pandera[io]` for
PyYAML) or **JSON**; `infer` can also emit **Python** modules. Data files can be
**Parquet**, **Feather**, **JSON**, and more for pandas-like backends.

For command options and limitations, see the {ref}`CLI reference <cli>`.
:::
