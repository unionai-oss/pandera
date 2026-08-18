"""Generate ``DataFrameModel`` class source from a :class:`DataFrameSchema`."""

from __future__ import annotations

import keyword
import re
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


def _dtype_to_simple_annotation(dtype: Any, backend: BackendId) -> str:
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
        # only the pandas backend has a dataframe-library-specific scalar
        # type for temporal columns; other backends use stdlib datetime.
        return "pd.Timestamp" if backend == "pandas" else "datetime.datetime"
    if "timedelta" in s or "duration" in s:
        return "pd.Timedelta" if backend == "pandas" else "datetime.timedelta"
    return "Any"


def _column_annotation(dtype: Any, backend: BackendId) -> str:
    inner = _dtype_to_simple_annotation(dtype, backend)
    if backend == "pandas":
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


def _format_field_from_column(
    column: Any, *, minimal: bool, alias: str | None = None
) -> str:
    """``Field(...)`` string for a schema column, or empty if defaults only."""
    parts: list[str] = []
    if alias is not None:
        parts.append(f"alias={alias!r}")
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


def _attribute_name_and_alias(col_name: Any) -> tuple[str, str | None]:
    """Class-attribute name for a column label, plus an ``alias`` if needed.

    Column labels are arbitrary (spaces, punctuation, non-strings), while
    model attributes must be Python identifiers. When they differ, the
    original label is preserved through ``Field(alias=...)``.
    """
    label = str(col_name)
    if label.isidentifier() and not keyword.iskeyword(label):
        return label, None
    sanitized = re.sub(r"\W", "_", label)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    if keyword.iskeyword(sanitized):
        sanitized = f"{sanitized}_"
    return sanitized, label


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

    col_lines: list[str] = []
    for col_name, column in dataframe_schema.columns.items():
        ann = _column_annotation(column.dtype, backend)
        # column labels that aren't valid Python identifiers can't be class
        # attribute names; emit a safe attribute name plus ``alias``.
        attr_name, alias = _attribute_name_and_alias(col_name)
        field_s = _format_field_from_column(
            column, minimal=minimal, alias=alias
        )
        if field_s:
            col_lines.append(f"    {attr_name}: {ann} = {field_s}")
        else:
            col_lines.append(f"    {attr_name}: {ann}")

    cfg = _config_class_body(dataframe_schema)
    fields_block = "\n".join(col_lines) if col_lines else "    pass"
    body = f"class {name}(pa.DataFrameModel):\n"
    if cfg:
        body += cfg
    body += fields_block + "\n"

    # only import what the annotations actually reference, so the emitted
    # script imports cleanly (and stays pandas-free for non-pandas backends).
    preamble = ""
    if "datetime." in body:
        preamble += "import datetime\n"
    if ": Any" in body or "[Any]" in body:
        preamble += "from typing import Any\n"
    if ("pd.Timestamp" in body or "pd.Timedelta" in body) and (
        "import pandas as pd" not in _BACKEND_IMPORTS[backend]
    ):
        preamble += "import pandas as pd\n"

    return f"{preamble}{_BACKEND_IMPORTS[backend]}\n\n{body}"
