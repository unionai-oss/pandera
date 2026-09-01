"""Rich (and plain) CLI reporting for validate, infer, and generate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors

if TYPE_CHECKING:
    from rich.console import ConsoleRenderable

MatchKey = tuple[Any, ...]


@dataclass(frozen=True)
class PlanRow:
    """One row in the validation checklist (schema-wide or column/index)."""

    group: str
    target: str
    requirement: str
    key: MatchKey


def _dtype_str(dtype: Any) -> str:
    if dtype is None:
        return "any"
    try:
        return str(dtype)
    except Exception:
        return type(dtype).__name__


def _check_label(check: Any) -> str:
    if check is None:
        return "check"
    name = getattr(check, "name", None)
    if callable(name):
        name = None
    if name:
        return str(name)
    fn = getattr(check, "fn", None)
    if fn is not None:
        return getattr(fn, "__name__", str(check))
    return str(check)


def _iter_index_schemas(schema: Any) -> list[tuple[str, Any]]:
    idx = getattr(schema, "index", None)
    if idx is None:
        return []
    if isinstance(idx, list):
        out: list[tuple[str, Any]] = []
        for i, ischema in enumerate(idx):
            nm = getattr(ischema, "name", None) or f"level {i}"
            out.append((str(nm), ischema))
        return out
    nm = getattr(idx, "name", None) or "index"
    return [(str(nm), idx)]


def _component_plan_rows(
    group_data: str,
    prefix: str,
    target: str,
    comp: Any,
) -> list[PlanRow]:
    rows: list[PlanRow] = []
    dt = getattr(comp, "dtype", None)
    if dt is not None:
        rows.append(
            PlanRow(
                group_data,
                target,
                f"dtype: {_dtype_str(dt)}",
                (prefix, target, "dtype"),
            )
        )
    if not getattr(comp, "nullable", True):
        rows.append(
            PlanRow(
                group_data,
                target,
                "non-null values",
                (prefix, target, "nullable"),
            )
        )
    if getattr(comp, "unique", False):
        rows.append(
            PlanRow(
                group_data,
                target,
                "unique values",
                (prefix, target, "unique"),
            )
        )
    for i, parser in enumerate(getattr(comp, "parsers", None) or []):
        rows.append(
            PlanRow(
                group_data,
                target,
                f"parser: {_check_label(parser)}",
                (prefix, target, "parser", i),
            )
        )
    for i, check in enumerate(getattr(comp, "checks", None) or []):
        rows.append(
            PlanRow(
                group_data,
                target,
                f"check: {_check_label(check)}",
                (prefix, target, "check", i),
            )
        )
    return rows


def build_validation_plan(schema: Any) -> tuple[list[PlanRow], frozenset[str]]:
    """Enumerate schema-wide and column/index validation steps."""
    rows: list[PlanRow] = []
    group_schema = "Schema (dataframe)"
    group_data = "Data (columns / index)"

    strict = getattr(schema, "strict", False)
    if strict:
        label = (
            "strict columns" if strict is True else "strict columns (filter)"
        )
        rows.append(PlanRow(group_schema, "—", label, ("df", "strict")))
    if getattr(schema, "ordered", False):
        rows.append(
            PlanRow(
                group_schema,
                "—",
                "column order",
                ("df", "ordered"),
            )
        )
    if getattr(schema, "unique_column_names", False):
        rows.append(
            PlanRow(
                group_schema,
                "—",
                "unique column names",
                ("df", "unique_colnames"),
            )
        )
    uq = getattr(schema, "unique", None)
    if uq:
        rows.append(
            PlanRow(
                group_schema,
                "—",
                "jointly unique column groups",
                ("df", "joint_unique"),
            )
        )
    if getattr(schema, "dtype", None) is not None:
        rows.append(
            PlanRow(
                group_schema,
                "—",
                f"dataframe dtype: {_dtype_str(schema.dtype)}",
                ("df", "dtype"),
            )
        )
    for i, check in enumerate(getattr(schema, "checks", None) or []):
        rows.append(
            PlanRow(
                group_schema,
                "—",
                f"check: {_check_label(check)}",
                ("df", "check", i),
            )
        )
    for i, parser in enumerate(getattr(schema, "parsers", None) or []):
        rows.append(
            PlanRow(
                group_schema,
                "—",
                f"parser: {_check_label(parser)}",
                ("df", "parser", i),
            )
        )

    index_targets: set[str] = set()
    for iname, ischema in _iter_index_schemas(schema):
        index_targets.add(iname)
        rows.extend(_component_plan_rows(group_data, "idx", iname, ischema))

    cols = getattr(schema, "columns", None) or {}
    for col_name, col_schema in cols.items():
        ckey = str(col_name)
        if getattr(col_schema, "required", True):
            rows.append(
                PlanRow(
                    group_data,
                    ckey,
                    "column present",
                    ("col", ckey, "presence"),
                )
            )
        rows.extend(_component_plan_rows(group_data, "col", ckey, col_schema))

    return rows, frozenset(index_targets)


def _reason_group(reason: SchemaErrorReason | None) -> str:
    if reason is None:
        return "other"
    if reason in (
        SchemaErrorReason.DATAFRAME_CHECK,
        SchemaErrorReason.DATAFRAME_PARSER,
        SchemaErrorReason.DUPLICATE_COLUMN_LABELS,
        SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
        SchemaErrorReason.COLUMN_NOT_ORDERED,
    ):
        return "schema"
    return "data"


def error_match_keys(
    err: SchemaError, index_targets: frozenset[str]
) -> set[MatchKey]:
    """Map a SchemaError to plan keys (best-effort)."""
    keys: set[MatchKey] = set()
    rc = err.reason_code
    col = err.column_name
    ci = err.check_index
    pi = err.parser_index
    chk = err.check

    def limb(name: str | None) -> str:
        if name is not None and name in index_targets:
            return "idx"
        return "col"

    if rc == SchemaErrorReason.DATAFRAME_CHECK:
        if ci is not None:
            keys.add(("df", "check", ci))
        return keys

    if rc == SchemaErrorReason.DATAFRAME_PARSER:
        if pi is not None:
            keys.add(("df", "parser", pi))
        return keys

    if rc == SchemaErrorReason.DUPLICATE_COLUMN_LABELS:
        keys.add(("df", "unique_colnames"))
        return keys

    if rc == SchemaErrorReason.COLUMN_NOT_IN_SCHEMA:
        keys.add(("df", "strict"))
        return keys

    if rc == SchemaErrorReason.COLUMN_NOT_ORDERED:
        keys.add(("df", "ordered"))
        return keys

    if rc == SchemaErrorReason.DUPLICATES and col is None:
        if chk == "multiple_fields_uniqueness":
            keys.add(("df", "joint_unique"))
        return keys

    if rc in (
        SchemaErrorReason.WRONG_DATATYPE,
        SchemaErrorReason.DATATYPE_COERCION,
    ):
        if col:
            keys.add((limb(col), col, "dtype"))
        else:
            keys.add(("df", "dtype"))
        return keys

    if rc == SchemaErrorReason.SERIES_CONTAINS_NULLS and col:
        keys.add((limb(col), col, "nullable"))
        return keys

    if rc in (
        SchemaErrorReason.DUPLICATES,
        SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
    ):
        if col:
            keys.add((limb(col), col, "unique"))
        return keys

    if rc == SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME and col:
        keys.add(("col", col, "presence"))
        return keys

    if rc == SchemaErrorReason.CHECK_ERROR:
        if col is not None and ci is not None:
            keys.add((limb(col), col, "check", ci))
        elif col is None and ci is not None:
            keys.add(("df", "check", ci))
        return keys

    if rc in (
        SchemaErrorReason.SCHEMA_COMPONENT_PARSER,
        SchemaErrorReason.PARSER_ERROR,
    ):
        if col is not None and pi is not None:
            keys.add((limb(col), col, "parser", pi))
        return keys

    if rc == SchemaErrorReason.MISMATCH_INDEX:
        keys.add(("idx", col or "index", "dtype"))
        return keys

    return keys


def _shorten_message(msg: str, max_len: int = 100) -> str:
    msg = " ".join(msg.split())
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 1] + "…"


def _format_failure_cases(failure_cases: Any) -> str:
    """Format :attr:`~pandera.errors.SchemaError.failure_cases` for the CLI."""
    if failure_cases is None:
        return "—"
    try:
        import pandas as pd

        if isinstance(failure_cases, pd.DataFrame):
            text = failure_cases.to_string(max_rows=10, max_cols=10)
            return _shorten_message(text, max_len=400)
        if isinstance(failure_cases, pd.Series):
            text = failure_cases.to_string(max_rows=15)
            return _shorten_message(text, max_len=400)
    except Exception:
        pass
    return _shorten_message(str(failure_cases), max_len=400)


def _normalize_errors(exc: SchemaError | SchemaErrors) -> list[SchemaError]:
    if isinstance(exc, SchemaErrors):
        return list(exc.schema_errors)
    return [exc]


def backend_label(backend_cls: type) -> str:
    """Short name of a validation backend class.

    ``pandera.backends.narwhals.container.DataFrameSchemaBackend`` →
    ``narwhals``; ``pandera.backends.pandas.container.DataFrameSchemaBackend``
    → ``pandas``. Falls back to the class name for non-pandera backends.
    """
    module = getattr(backend_cls, "__module__", "") or ""
    prefix = "pandera.backends."
    if module.startswith(prefix):
        return module[len(prefix) :].split(".", 1)[0]
    return backend_cls.__name__


def _console():
    from rich.console import Console

    return Console(soft_wrap=True)


def _backend_meta(backend_name: str | None) -> Any | None:
    """A small key/value table showing the validation backend in use."""
    if backend_name is None:
        return None
    from rich.table import Table

    meta = Table(show_header=False, box=None, padding=(0, 2))
    meta.add_row("[bold]Backend[/bold]", backend_name)
    return meta


def _panel_body(*parts: Any) -> ConsoleRenderable:
    """A single renderable, or a ``Group`` when there are several parts."""
    if len(parts) == 1:
        return parts[0]
    from rich.console import Group

    return Group(*parts)


def print_validation_success(
    schema: Any, backend_name: str | None = None
) -> None:
    plan, _ = build_validation_plan(schema)
    title = "[bold green]Validation succeeded[/bold green]"
    subtitle = "All listed schema- and data-level requirements passed."
    try:
        from rich.panel import Panel
        from rich.table import Table

        console = _console()
        table = Table(
            title="Checks",
            show_header=True,
            header_style="bold",
            expand=True,
        )
        table.add_column("Level")
        table.add_column("Target")
        table.add_column("Requirement")
        table.add_column("Status", justify="center")
        for row in plan:
            g = "schema" if row.group.startswith("Schema") else "data"
            table.add_row(
                g,
                row.target,
                row.requirement,
                "[green]passed[/green]",
            )
        body: list[Any] = []
        meta = _backend_meta(backend_name)
        if meta is not None:
            body.append(meta)
        body.append(table)
        console.print(
            Panel(
                _panel_body(*body),
                title=title,
                subtitle=subtitle,
                border_style="green",
                expand=True,
            )
        )
    except ImportError:
        print("Validation succeeded.")
        if backend_name is not None:
            print(f"  Backend: {backend_name}")
        for row in plan:
            scope = "schema" if row.group.startswith("Schema") else "data"
            print(f"  [{scope}] {row.target}: {row.requirement} — passed")


def print_validation_failure(
    schema: Any,
    exc: SchemaError | SchemaErrors,
    backend_name: str | None = None,
) -> None:
    plan, index_targets = build_validation_plan(schema)
    errors = _normalize_errors(exc)
    failed_keys: set[MatchKey] = set()
    for err in errors:
        failed_keys.update(error_match_keys(err, index_targets))

    try:
        from rich.panel import Panel
        from rich.table import Table

        console = _console()
        main = Table(
            title="Check results",
            show_header=True,
            header_style="bold",
            expand=True,
        )
        main.add_column("Level")
        main.add_column("Target")
        main.add_column("Requirement")
        main.add_column("Status", justify="center")
        for row in plan:
            lvl = "schema" if row.group.startswith("Schema") else "data"
            st = (
                "[red]failed[/red]"
                if row.key in failed_keys
                else "[green]passed[/green]"
            )
            main.add_row(lvl, row.target, row.requirement, st)

        fail_tbl = Table(
            title="Failure details",
            show_header=True,
            header_style="bold",
            expand=True,
        )
        fail_tbl.add_column("Level")
        fail_tbl.add_column("Target")
        fail_tbl.add_column("Reason")
        fail_tbl.add_column("Failure cases", overflow="fold")
        fail_tbl.add_column("Message", overflow="fold")
        for err in errors:
            tgt = err.column_name if err.column_name is not None else "—"
            rc = err.reason_code.value if err.reason_code else "—"
            fail_tbl.add_row(
                _reason_group(err.reason_code),
                str(tgt),
                rc,
                _format_failure_cases(err.failure_cases),
                _shorten_message(str(err)),
            )
        body: list[Any] = []
        meta = _backend_meta(backend_name)
        if meta is not None:
            body.append(meta)
        body.append(main)
        console.print(
            Panel(
                _panel_body(*body),
                title="[bold red]Validation failed[/bold red]",
                border_style="red",
                expand=True,
            )
        )
        console.print(
            Panel(fail_tbl, border_style="red", expand=True),
        )
    except ImportError:
        print("Validation failed.")
        if backend_name is not None:
            print(f"  Backend: {backend_name}")
        for row in plan:
            lvl = "schema" if row.group.startswith("Schema") else "data"
            st = "failed" if row.key in failed_keys else "passed"
            print(f"  [{lvl}] {row.target}: {row.requirement} — {st}")
        for err in errors:
            print(f"  ! {_shorten_message(str(err))}")
            fc = _format_failure_cases(err.failure_cases)
            if fc != "—":
                print(f"    failure cases: {fc}")


def _file_brief(path: Path) -> str:
    try:
        st = path.stat()
    except OSError:
        return f"{path} (unreadable)"
    return f"{path} ({st.st_size} bytes)"


def _dataset_shape_text(obj: Any) -> str:
    h = getattr(obj, "height", None)
    w = getattr(obj, "width", None)
    if h is not None and w is not None:
        return f"{h} rows × {w} columns"
    shp = getattr(obj, "shape", None)
    if shp is not None:
        try:
            if len(shp) >= 2:
                return f"{shp[0]} rows × {shp[1]} columns"
        except Exception:
            pass
    nrows = getattr(obj, "n_rows", None)
    ncols = getattr(obj, "n_columns", None)
    if nrows is not None and ncols is not None:
        return f"{nrows} rows × {ncols} columns"
    return "unknown"


def _schema_columns_preview(
    schema: Any, limit: int = 30
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    cols = getattr(schema, "columns", None) or {}
    for i, (name, col) in enumerate(cols.items()):
        if i >= limit:
            rows.append(("…", f"({len(cols) - limit} more)"))
            break
        rows.append((str(name), _dtype_str(getattr(col, "dtype", None))))
    return rows


def print_infer_summary(
    *,
    data_path: Path,
    backend: str,
    obj: Any,
    schema: Any,
    output: Path,
    fmt: str,
) -> None:
    try:
        from rich.panel import Panel
        from rich.table import Table

        console = _console()
        meta = Table(show_header=False, box=None, padding=(0, 2))
        meta.add_row("Data file", _file_brief(data_path))
        meta.add_row("Backend", backend)
        meta.add_row("Shape", _dataset_shape_text(obj))
        meta.add_row("Output", str(output))
        meta.add_row("Format", fmt)
        console.print(
            Panel.fit(
                meta,
                title="[bold cyan]Infer[/bold cyan]",
                border_style="cyan",
            )
        )
        st = Table(title="Inferred columns", header_style="bold")
        st.add_column("Column")
        st.add_column("dtype")
        for name, dt in _schema_columns_preview(schema):
            st.add_row(name, dt)
        console.print(st)
        console.print(
            f"[green]Wrote inferred schema to[/green] [bold]{output}[/bold]"
        )
    except ImportError:
        print(f"Wrote inferred schema to {output}")


def _pandas_schema_brief(schema: Any) -> str:
    cols = getattr(schema, "columns", None) or {}
    return f"{len(cols)} column(s)"


def _xarray_schema_brief(schema: Any) -> str:
    dvars = getattr(schema, "data_vars", None)
    if dvars is not None:
        coords = getattr(schema, "coords", None) or {}
        return f"{len(dvars)} data_var(s), {len(coords)} coord(s)"
    dims = getattr(schema, "dims", None)
    if dims is not None:
        return f"DataArray, dims {tuple(dims)!r}"
    return type(schema).__name__


def _generated_data_summary(obj: Any, schema_kind: str) -> str:
    if schema_kind == "pandas":
        shp = getattr(obj, "shape", None)
        if shp is not None:
            return f"DataFrame shape {shp[0]} × {shp[1]}"
        return type(obj).__name__
    try:
        import xarray as xr

        if isinstance(obj, xr.Dataset):
            return f"Dataset dims {dict(obj.sizes)}"
        if isinstance(obj, xr.DataArray):
            return f"DataArray dims {dict(obj.sizes)}"
    except ImportError:
        pass
    return type(obj).__name__


def print_generate_summary(
    *,
    schema_path: Path,
    schema_kind: str,
    schema_obj: Any,
    size: int,
    output: Path,
    writer_key: str,
    data_obj: Any,
) -> None:
    try:
        from rich.panel import Panel
        from rich.table import Table

        console = _console()
        sch_txt = (
            _pandas_schema_brief(schema_obj)
            if schema_kind == "pandas"
            else _xarray_schema_brief(schema_obj)
        )
        meta = Table(show_header=False, box=None, padding=(0, 2))
        meta.add_row("Schema file", str(schema_path))
        meta.add_row("Schema kind", schema_kind)
        meta.add_row("Schema", sch_txt)
        meta.add_row("Requested size", str(size))
        meta.add_row("Output", str(output))
        meta.add_row("Writer", writer_key)
        console.print(
            Panel.fit(
                meta,
                title="[bold magenta]Generate[/bold magenta]",
                border_style="magenta",
            )
        )
        console.print(
            Panel.fit(
                f"[bold]Generated[/bold]\n{_generated_data_summary(data_obj, schema_kind)}",
                border_style="magenta",
            )
        )
        console.print(
            f"[green]Wrote generated data to[/green] [bold]{output}[/bold]"
        )
    except ImportError:
        print(f"Wrote generated data to {output}")
