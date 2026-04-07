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
pandera infer -d /tmp/dataset.csv -o /tmp/schema.json
```

You should see a short summary (paths, shape, columns) and a line that the
schema was written successfully.

## Validate the good data

```bash
pandera validate -s /tmp/schema.json -d /tmp/dataset.csv
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
pandera validate -s /tmp/schema.json -d /tmp/invalid_dataset.csv
```

This should exit with a **non-zero** status. You should see **Validation failed**
on standard error plus tables listing which checks passed or failed and failure
details (exact layout depends on your Pandera and Rich versions).

## Generate synthetic data from the schema

Create a new CSV of synthetic rows that satisfy the schema:

```bash
pandera generate -s /tmp/schema.json -o /tmp/generated.csv --size 5
```

Open `/tmp/generated.csv` to inspect the generated rows.

:::{note}
**Other backends and formats**

The same CLI supports **Polars**, **PySpark**, **Ibis**, and other loaders
depending on how you install Pandera and which `--backend` you pass to
`infer` / `validate`. Schema files can be **YAML** (install `pandera[io]` for
PyYAML) or **JSON**; `infer` can also emit **Python** modules. Data files can be
**Parquet**, **Feather**, **JSON**, and more for pandas-like backends.

For command options and limitations, see the {ref}`CLI reference <cli>`.
:::
