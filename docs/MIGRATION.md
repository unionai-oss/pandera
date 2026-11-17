# Documentation migration: Sphinx (Furo) → Material for MkDocs

Pandera's documentation was migrated from **Sphinx + the Furo theme** to
**Material for MkDocs** — the same documentation stack that
[zensical.org](https://zensical.org/) is built on (its creators also maintain
Material for MkDocs; the theme is our configured `mkdocs.yml` `theme`). This
README is the migration record: it documents the URL strategy, the redirect
registry, and how to build and maintain the site.

## Where things live

| Path | Purpose |
| --- | --- |
| `mkdocs.yml` | MkDocs configuration (Material theme, plugins, redirects). Read by Read the Docs. |
| `mkdocs/` | **New** MkDocs source tree (converted pages, reference section pages). |
| `mkdocs/gen_ref_pages.py` | Generates the per-object API reference pages under `mkdocs/reference/generated/` from `mkdocs.yml`'s object catalogue (run by the `gen-files` plugin at build time). |
| `mkdocs/reference/*.md` | Hand-authored API section landing pages (Core, Dtypes, Errors, …) linking to the generated detail pages. |
| `.readthedocs.yaml` | Read the Docs v2 build config. |
| `docs/source/` | **Old** Sphinx source. Kept during the migration for history / rollback. Build tooling (`docs/Makefile`) is unchanged and still works. |

## URL strategy (why most links keep working)

Read the Docs serves the built site under `/en/<version>/`. Both Sphinx and
MkDocs produce `<page>.html` slugs, and this repo keeps **the same relative
path for every page**:

- `docs/source/dataframe_schemas.md` → `mkdocs/dataframe_schemas.md` →
  `/en/stable/dataframe_schemas.html` (identical to the old URL).
- API detail pages: the old Sphinx `autosummary` produced
  `reference/generated/<fully.qualified.name>.html`. `mkdocs/gen_ref_pages.py`
  reproduces those exact paths, so e.g.
  `reference/generated/pandera.api.pandas.container.DataFrameSchema.html` is
  unchanged.
- Guide subdirectories (`xarray_guide/`, `pytorch_guide/`) keep their paths.

`mkdocs.yml` sets `use_directory_urls: false` so pages are emitted as
`foo.html` rather than `foo/`, matching the historical layout.

## Redirect registry

Read the Docs has **no config-file mechanism** for redirects (they are only
manageable through its web UI or the API). Instead, this repo uses the
**`mkdocs-redirects` plugin**, which bakes real redirect pages into the built
site — fully declarative, reviewable, and deployable from the repo without
touching the Read the Docs admin console.

Current redirects (defined in `mkdocs.yml` → `plugins.redirects.redirect_maps`.
Keys are `.md` source paths; the plugin emits the `.html` page name):

> `build --strict` note: MkDocs currently warns “Doc file … contains a link …
> not found among documentation files” for every internal `.html` link. These
> are false positives — MkDocs’s link validator compares `.html` destinations to
> the `.md` sources and ignores `use_directory_urls: false`. Every such link
> resolves to a real page in the built site (verified independently during the
> migration). Keeping `fail_on_warning: false` is required to preserve the
> `.html` URL slugs, which is the point of this migration.

| From | To | Reason |
| --- | --- | --- |
| `genindex.html` | `index.html` | Sphinx auto-generated "index" page; MkDocs has no equivalent. |
| `py-modindex.html` | `index.html` | Sphinx Python module index; MkDocs has no equivalent. |
| `search.html` | `index.html` | Sphinx generated a search page; Material's search is a modal overlay. |

Everything else maps 1:1 onto the old site, so no further redirects are
required. Two non-page artifacts are intentionally *not* redirected:

- `objects.inv` (the Sphinx intersphinx inventory) is not emitted by MkDocs.
  External projects that `intersphinx` resolved `pandera` symbols against that
  inventory; if any still do, point them at the MkDocs API pages instead.
- `_static/*` (Sphinx theme assets) are gone; pandera's branding images now
  live under `mkdocs/css/`.

## Executing code cells

The old ``myst-nb`` build executed every ``{code-cell}`` block during the docs
build and rendered its output. The Material-for-MkDocs build restores this with
a small in-tree MkDocs plugin, ``pandera/_docs/mkdocs_exec.py`` (registered as
the ``exec-codes`` plugin in ``mkdocs.yml``).

- Cells are identified **without markers in the fence language info** (Material's
  highlighter cannot parse a free second token like ``exec``). Instead, the plugin
  reads the original ``docs/source/*.md`` files, extracts the ordered
  ``{code-cell}`` bodies, and matches each page's python fences to them by body.
  This keeps ``docs/source`` the single source of truth for what executes and
  requires no special fence syntax in ``mkdocs/``.
- Each page executes in an isolated module registered in ``sys.modules`` so
  ``typing.get_type_hints`` resolves class-based ``DataFrameModel`` annotations
  (the same reason IPython's user namespace is ``__main__``).
- Output (stdout + the trailing expression) is injected as plain-text ``<pre>``
  blocks styled by ``.doc-exec-*`` classes — never raw HTML tables, which would
  break surrounding Markdown structure.
- Cells tagged ``raises-exception`` in the source render their traceback instead
  of failing (these are the deliberate "Bad" examples in the docs).
- ``plugins.exec-codes.strict`` defaults to ``false``: a cell that raises in a
  given build environment renders its traceback inline rather than aborting the
  build, so Read the Docs always publishes even if a backend is unavailable in an
  edge environment. Set it to ``true`` once the full cell suite has been verified
  against the exact build environment.

To run the docs (including cell execution) locally:

```bash
uv venv .venv-mkdocs
uv pip install --python .venv-mkdocs/bin/python -e ".[all,docs-mkdocs]"
.venv-mkdocs/bin/python -m mkdocs build -f mkdocs.yml
```

## How to build

A quick build command:

```bash
mkdir -p site && .venv-mkdocs/bin/python -m mkdocs build -f mkdocs.yml
```

## Editing the API reference

- To add/remove/reslug an object, edit the per-section catalogue in
  `mkdocs/gen_ref_pages.py` (the `GROUPS` dict). The section landing pages in
  `mkdocs/reference/*.md` link to the generated detail pages.
- Rebuild with the command above; the `gen-files` plugin regenerates the
  detail pages under `mkdocs/reference/generated/` (that directory is
  git-ignored because it is produced at build time).

## Read the Docs

`.readthedocs.yaml` configures RTD to build the MkDocs site with `mkdocs.yml`,
installing pandera's `all` extra plus `docs-mkdocs`. `fail_on_warning` is left
`false`; set it to `true` once links are fully clean.
