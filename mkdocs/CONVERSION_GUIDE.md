# Sphinx → Material for MkDocs conversion guide

This repo is migrating its documentation from Sphinx (Furo) to Material for
MkDocs (the same stack zensical.org is built on). Source Markdown currently
lives in `docs/source/*.md` (MyST-flavoured Markdown). The conversion target
lives in `mkdocs/*.md`. **Preserve the URL slug**: every file must keep the
same path relative to `docs/source/` (so `docs/source/dataframe_models.md`
becomes `mkdocs/dataframe_models.md` and is served at `dataframe_models.html`).

Follow these rules. When in doubt, look at `mkdocs/index.md` — it is a fully
converted exemplar.

## 1. Front matter & document setup

- Drop the MyST front matter block (`---\nfile_format: mystnb\n---`) and any
  `%` comment lines and `(label)=` anchor denominator lines. Keep a `#` title.
- Drop Sphinx-only sections at the bottom of files: ```` ```{toctree} ... ``` ````,
  `# Indices and tables`, `- {ref}\`genindex\``.
- A file's own heading determines the auto-anchor used by in-page links, so
  map `{ref}` labels pointing at that page to that heading's slug.

## 2. Images and badges

- ```` ```{image} URL\n:alt: ALT\n:target: URL2\n``` ```` becomes
  `[![ALT](IMG)](TARGET)`. If no `:target:`, use `![ALT](IMG)`.
- Badge shields go inline; do not wrap each in a div.

## 3. Admonitions

- `:::{note}` … `:::` → `!!! note` … (indented body 4 spaces).
- `:::{warning}` → `!!! warning`; `:::{important}` → `!!! important`;
  `:::{tip}` → `!!! tip`; `:::{danger}` → `!!! danger`.
- `:::{admonition} Title\n:class: important` → `!!! important "Title"`.
- `:::{attention}` → `!!! warning "Attention"`.

## 4. Tabs

MyST tab-set/tab-item (`::::{tab-set}` `:::{tab-item} pip`) → MkDocs tabbed
syntax:

```
=== "pip"

    ```bash
    ...
    ```

=== "conda"

    ```bash
    ...
    ```
```

## 5. Code blocks / code-cells

- ```` ```{code-cell} python\n:tags: [...]\n```` → a plain ````python` fence,
  dropping the `:tags:`/`:name:` metadata lines entirely.
- ```` ```{code} bash ```` → ````bash` fence.

## 6. Cross-references

### 6a. `{ref}` labels

`{ref}\`text <label>\`` → `[text](target.html)` (or `target.html#anchor` for a
known sub-heading). `{ref}\`label\`` → `[the target's title](target.html)`.

Mapping of the common labels (labels not listed: map by obvious page name and
verify against the file list):

| label | target |
| --- | --- |
| `installation` | `index.html#install` |
| `supported-dataframe-libraries`, `dataframe-libraries`, `supported-features` | `supported_libraries.html` |
| `dataframeschemas`, `dataframe-schemas` | `dataframe_schemas.html` |
| `dataframe-models` | `dataframe_models.html` |
| `seriesschemas` | `series_schemas.html` |
| `dtype-validation` | `dtype_validation.html` |
| `checks` | `checks.html` |
| `dtypes`, `parameterized-dtypes` | `dtypes.html` |
| `lazy-validation` | `lazy_validation.html` |
| `error-report` | `error_report.html` |
| `cli` | `cli.html` |
| `financial-reconciliation` | `financial_reconciliation.html` |
| `decision-parsing-tutorial` | `decision_parsing_tutorial.html` |
| `decisions` | `decisions.html` |
| `hypothesis` | `hypothesis.html` |
| `parsers` | `parsers.html` |
| `derived-columns` | `parsers.html#derived-columns` |
| `decorators` | `decorators.html` |
| `drop-invalid-rows` | `drop_invalid_rows.html` |
| `schema-inference` | `schema_inference.html` |
| `data-synthesis-strategies` | `data_synthesis_strategies.html` |
| `data-format-conversion` | `data_format_conversion.html` |
| `extensions` | `extensions.html` |
| `configuration` | `configuration.html` |
| `schema-model-config` | `configuration.html#schema-model-config` |
| `adding-missing-columns`, `column-check-groups`, `index-validation` | `dataframe_schemas.html` (anchor if present as a heading) |
| `integrations` | `integrations.html` |
| `pyarrow` | `pyarrow.html` |
| `datafusion` | `datafusion.html` |
| `polars` | `polars.html` |
| `ibis` | `ibis.html` |
| `native-pyspark` | `pyspark.html` |
| `pyspark-sql` | `pyspark_sql.html` |
| `narwhals-backend`, `narwhals-backend-registration`, `narwhals-opting-out` | `narwhals_backend.html` |
| `schema-persistence` | `frictionless.html` |
| `dask` | `dask.html` |
| `modin` | `modin.html` |
| `geopandas` | `geopandas.html` |
| `fastapi-integration` | `fastapi_integration.html` |
| `pydantic-integration` | `pydantic_integration.html` |
| `mypy` | `mypy_integration.html` |
| `xarray-guide` | `xarray_guide/index.html` |
| `pytorch-guide` | `pytorch_guide/index.html` |
| `api-core` | `reference/core.html` |
| `api-dataframe-models` | `reference/dataframe_models.html` |
| `api-geopandas` | `reference/geopandas.html` |
| `api-dtypes` | `reference/dtypes.html` |
| `api-decorators` | `reference/decorators.html` |
| `api-schema-inference` | `reference/schema_inference.html` |
| `api-io-utils` | `reference/io.html` |
| `api-strategies` | `reference/strategies.html` |
| `api-narwhals` | `reference/narwhals.html` |
| `api-parsers` | `reference/parsers.html` |
| `api-decisions` | `reference/decisions.html` |
| `api-extensions` | `reference/extensions.html` |
| `api-errors` | `reference/errors.html` |
| `api-xarray` | `reference/xarray.html` |
| `api-pytorch` | `reference/pytorch.html` |

For `xarray_guide/*` anchors (e.g. `xarray-dataset-schema`) and
`pytorch_guide/*` anchors (e.g. `pytorch-tensordict-schema`), link to the
matching `xarray_guide/<page>.html` / `pytorch_guide/<page>.html` with the
same `#slug` where the heading exists.

### 6b. `{class}` / `{func}` / `{meth}` / `{attr}` / `{data}` API crossrefs

`{class}\`~pandera.api.pandas.container.DataFrameSchema\`` →
`[DataFrameSchema](reference/generated/pandera.api.pandas.container.DataFrameSchema.html)`.
Rules:

- Strip the leading `~`. The link text is the last dotted segment.
- `pandera.set_config` → `reference/generated/pandera.set_config.html`.
- **Methods** and members (a dotted path longer than the documented object)
  link to their **owning object's** generated page. E.g.
  `~pandera.api.pandas.model.DataFrameModel.validate` →
  `reference/generated/pandera.api.pandas.model.DataFrameModel.html`.
  Same for `...validate`, `...to_schema`, `...from_json`, `...schema_errors`,
  `...set_index`, `...update_column`, `...update_index`, `...example`, `...empty`.
- Typos corrected to the documented object:
  `~pandera.pandas_engine.PydanticModel` → `reference/generated/pandera.engines.pandas_engine.PydanticModel.html`.
- External types (`~xarray.DataArray`, `~tensordict.TensorDict`,
  `typing.Annotated`, `typing.Union`, `classmethod`) render as **inline code**
  (no internal link). Link `xarray.*` to
  `https://docs.xarray.dev/en/stable/` if a natural anchor is obvious,
  otherwise inline code.

## 7. Math

- Sphinx `.. math::` directives and `:math:` roles → `$$ ... $$` (block) or
  `$ ... $` (inline).

## 8. Tables

- `::::{table}` / `:::{table}` directives → plain GitHub-flavoured Markdown
  tables. `:widths:` and `:align:` are dropped.
- `.. list-table::` (in RST) → Markdown table or definition list.

## 9. Literal / eval-rst

- ```` ```{eval-rst} ... ```` blocks publish raw reStructuredText; if the RST
  is a `list-table`/`currentmodule`, convert to Markdown by hand. Never leave
  raw RST in the output.

## 10. Do NOT touch

- Do not edit `mkdocs.yml`, `mkdocs/index.md`, `mkdocs/gen_ref_pages.py`, the
  `reference/*.md` files, or your assigned files' siblings outside the list in
  your task.
- Do not run `mkdocs build` and do not `git add/commit` unless your task says
  you may. Write the converted `.md` files only.

## Verification

After converting every file in your list, print a short summary: the list of
files written and any labels/roles you could not confidently map (so they can
be reviewed).
