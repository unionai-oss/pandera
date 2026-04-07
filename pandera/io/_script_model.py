"""Generate ``DataFrameModel`` class source from a :class:`DataFrameSchema`."""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pandera.api.checks import Check
from pandera.io._minimal import COLUMN_DEFAULTS, DF_SCHEMA_DEFAULTS

BackendId = Literal["pandas", "polars", "ibis", "pyspark"]

# Check.name (builtin) -> Field keyword on :func:`~pandera.api.dataframe.model_components.Field`.
_CHECK_NAME_TO_FIELD: dict[str, str] = {
    "equal_to": "eq",
    "not_equal_to": "ne",
    "greater_than": "gt",
    "greater_than_or_equal_to": "ge",
    "less_than": "lt",
    "less_than_or_equal_to": "le",
    "in_range": "in_range",
    "between": "between",
    "isin": "isin",
    "notin": "notin",
    "str_contains": "str_contains",
    "str_endswith": "str_endswith",
    "str_matches": "str_matches",
    "str_length": "str_length",
    "str_startswith": "str_startswith",
    "unique_values_eq": "unique_values_eq",
}

_BACKEND_IMPORTS: dict[BackendId, str] = {
    "pandas": (
        "import pandas as pd\n"
        "import pandera.pandas as pa\n"
        "from pandera.typing import Series\n"
    ),
    "polars": ("import pandera.polars as pa\n"),
    "ibis": ("import pandera.ibis as pa\n"),
    "pyspark": ("import pandera.pyspark as pa\n"),
}


def _dtype_to_simple_annotation(dtype: Any) -> str:
    """Map an engine dtype to a simple typing annotation string."""
    if dtype is None:
        return "Any"
    s = str(dtype).lower()
    if "int" in s:
        return "int"
    if "float" in s or "double" in s or "decimal" in s:
        return "float"
    if "bool" in s:
        return "bool"
    if (
        "str" in s
        or "string" in s
        or "utf8" in s
        or "object" in s
        or "binary" in s
    ):
        return "str"
    if "datetime" in s or "timestamp" in s or "date" in s:
        return "pd.Timestamp"
    if "timedelta" in s:
        return "pd.Timedelta"
    if "category" in s:
        return "Any"
    return "Any"


def _column_annotation(dtype: Any, backend: BackendId) -> str:
    inner = _dtype_to_simple_annotation(dtype)
    if backend == "pandas":
        if inner == "pd.Timestamp":
            return "Series[pd.Timestamp]"
        if inner == "pd.Timedelta":
            return "Series[pd.Timedelta]"
        if inner == "Any":
            return "Series[Any]"
        return f"Series[{inner}]"
    return inner


def _in_range_field_value(stats: dict[str, Any]) -> str:
    keys = {"min_value", "max_value", "include_min", "include_max"}
    if not stats:
        return "{}"
    if set(stats.keys()) <= keys:
        mv = stats.get("min_value")
        xv = stats.get("max_value")
        imn = stats.get("include_min", True)
        imx = stats.get("include_max", True)
        if imn is True and imx is True:
            return f"({mv!r}, {xv!r})"
    return repr(stats)


def _str_length_field_value(stats: dict[str, Any]) -> str:
    ev = stats.get("exact_value")
    if ev is not None:
        return repr(ev)
    mv = stats.get("min_value")
    xv = stats.get("max_value")
    if mv is not None and xv is not None:
        return f"({mv!r}, {xv!r})"
    if mv is not None:
        return f"({mv!r},)"
    if xv is not None:
        return f"(None, {xv!r})"
    return repr(stats)


def _field_kwarg_for_check(check: Check) -> tuple[str, str] | None:
    """One ``Field`` keyword from a :class:`~pandera.api.checks.Check`."""
    fname = _CHECK_NAME_TO_FIELD.get(check.name)  # type: ignore[arg-type]
    if fname is None:
        warnings.warn(
            f"Check `{check.name}` has no Field() equivalent; skipped in model script.",
            UserWarning,
            stacklevel=3,
        )
        return None
    st = dict(check.statistics) if check.statistics else {}

    if check.name == "in_range":
        return fname, _in_range_field_value(st)
    if check.name == "str_length":
        return fname, _str_length_field_value(st)
    if check.name == "between":
        return fname, repr(st)
    if check.name in ("isin", "notin"):
        vals = st.get("allowed_values")
        if vals is None:
            vals = st.get("values")
        return fname, repr(vals)
    if check.name in (
        "str_contains",
        "str_endswith",
        "str_matches",
        "str_startswith",
    ):
        return fname, repr(st.get("pattern"))
    if check.name in ("equal_to", "not_equal_to"):
        return fname, repr(st.get("value"))
    if check.name in ("greater_than", "greater_than_or_equal_to"):
        return fname, repr(st.get("min_value"))
    if check.name in ("less_than", "less_than_or_equal_to"):
        return fname, repr(st.get("max_value"))
    if check.name == "unique_values_eq":
        return fname, repr(st.get("values"))

    return fname, repr(st)


def _format_field_from_column(column: Any, *, minimal: bool) -> str:
    """``Field(...)`` string for a schema column, or empty if defaults only."""
    parts: list[str] = []
    for check in column.checks or []:
        try:
            in_registry = check in Check
        except TypeError:
            in_registry = False
        if not in_registry:
            warnings.warn(
                f"Check `{getattr(check, 'name', check)}` is not a registered "
                "builtin; skipped in model script.",
                UserWarning,
                stacklevel=3,
            )
            continue
        kw = _field_kwarg_for_check(check)
        if kw:
            parts.append(f"{kw[0]}={kw[1]}")

    for attr in (
        "nullable",
        "unique",
        "coerce",
        "required",
        "regex",
        "title",
        "description",
    ):
        if not hasattr(column, attr):
            continue
        val = getattr(column, attr)
        default = COLUMN_DEFAULTS.get(attr)
        if minimal and default is not None and val == default:
            continue
        if val is None and attr in ("title", "description"):
            continue
        parts.append(f"{attr}={val!r}")

    if not parts:
        return ""
    inner = ", ".join(parts)
    return f"pa.Field({inner})"


def _config_class_body(schema: Any) -> str:
    lines: list[str] = []
    for key in sorted(DF_SCHEMA_DEFAULTS.keys()):
        if not hasattr(schema, key):
            continue
        val = getattr(schema, key)
        if val == DF_SCHEMA_DEFAULTS[key]:
            continue
        lines.append(f"        {key} = {val!r}")
    if not lines:
        return ""
    return "\n    class Config:\n" + "\n".join(lines) + "\n"


def to_dataframe_model_script(
    dataframe_schema: Any,
    backend: BackendId,
    *,
    minimal: bool = True,
    class_name: str | None = None,
) -> str:
    """Return Python source defining a :class:`DataFrameModel` subclass."""
    name = class_name or (
        dataframe_schema.name if dataframe_schema.name else "GeneratedModel"
    )
    if not name.isidentifier():
        name = "GeneratedModel"

    lines: list[str] = [_BACKEND_IMPORTS[backend], "\n\n"]

    col_lines: list[str] = []
    for col_name, column in dataframe_schema.columns.items():
        ann = _column_annotation(column.dtype, backend)
        field_s = _format_field_from_column(column, minimal=minimal)
        if field_s:
            col_lines.append(f"    {col_name}: {ann} = {field_s}")
        else:
            col_lines.append(f"    {col_name}: {ann}")

    cfg = _config_class_body(dataframe_schema)
    fields_block = "\n".join(col_lines) if col_lines else "    pass"
    lines.append(f"class {name}(pa.DataFrameModel):\n")
    if cfg:
        lines.append(cfg)
    lines.append(fields_block)
    lines.append("\n")

    script = "".join(lines)
    if backend == "pandas" and "pd.Timestamp" in script:
        script = "from pandas import Timestamp\n" + script
    if backend == "pandas" and "pd.Timedelta" in script:
        script = "from pandas import Timedelta\n" + script
    if "pd.Timestamp" in script or "pd.Timedelta" in script:
        if "import pandas as pd" not in script:
            script = "import pandas as pd\n" + script

    return script
