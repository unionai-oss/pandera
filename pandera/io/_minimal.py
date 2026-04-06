"""Strip serialized schema dicts down to non-default options (``minimal`` mode)."""

from __future__ import annotations

from typing import Any

from pandera.api.checks import Check

# --- DataFrameSchema (api.dataframe.container.DataFrameSchema) -------------
DF_SCHEMA_DEFAULTS: dict[str, Any] = {
    "dtype": None,
    "coerce": False,
    "strict": False,
    "name": None,
    "ordered": False,
    "unique": None,
    "report_duplicates": "all",
    "unique_column_names": False,
    "add_missing_columns": False,
    "title": None,
    "description": None,
}

# Column / shared component (pandas Column; polars/ibis/pyspark overlap)
COLUMN_DEFAULTS: dict[str, Any] = {
    "nullable": False,
    "unique": False,
    "report_duplicates": "all",
    "coerce": False,
    "required": True,
    "regex": False,
    "title": None,
    "description": None,
    "default": None,
    "drop_invalid_rows": False,
    "name": None,
}

# Check.options keys that mirror Check.__init__ defaults
CHECK_OPTION_DEFAULTS: dict[str, Any] = {
    "ignore_na": True,
    "raise_warning": False,
}


def _prune_check_options_dict(options: dict[str, Any]) -> None:
    for key in list(options.keys()):
        if key == "check_name":
            continue
        if key in CHECK_OPTION_DEFAULTS:
            if options[key] == CHECK_OPTION_DEFAULTS[key]:
                del options[key]


def prune_serialized_check_entry(entry: Any, check_obj: Check | None) -> None:
    """Prune default check metadata from one serialized check dict."""
    if not isinstance(entry, dict) or check_obj is None:
        return
    opts = entry.get("options")
    if isinstance(opts, dict):
        _prune_check_options_dict(opts)
        if not opts:
            entry.pop("options", None)
        elif opts.keys() == {"check_name"}:
            pass


def _prune_check_list(
    serialized: list[Any] | None, checks: list[Any] | None
) -> None:
    if not serialized or not checks:
        return
    for i, ser in enumerate(serialized):
        if i < len(checks) and isinstance(checks[i], Check):
            prune_serialized_check_entry(ser, checks[i])


def prune_component_dict(ser: dict[str, Any], component: Any) -> None:
    """Remove component keys whose values match constructor defaults."""
    if getattr(component, "dtype", None) is None:
        ser.pop("dtype", None)

    ch = getattr(component, "checks", None)
    if not ch:
        ser.pop("checks", None)
    elif isinstance(ser.get("checks"), list):
        _prune_check_list(ser["checks"], list(ch))

    for key in list(ser.keys()):
        if key in ("dtype", "checks"):
            continue
        if key not in COLUMN_DEFAULTS:
            continue
        if not hasattr(component, key):
            continue
        if getattr(component, key) == COLUMN_DEFAULTS[key]:
            ser.pop(key, None)


def _index_components(schema: Any) -> list[Any]:
    idx = getattr(schema, "index", None)
    if idx is None:
        return []
    try:
        return list(idx.indexes)
    except AttributeError:
        return [idx]


def apply_minimal_dataframe_container(
    out: dict[str, Any], schema: Any
) -> None:
    """In-place: drop keys matching :class:`DataFrameSchema` defaults."""
    out.pop("version", None)

    for key, default in DF_SCHEMA_DEFAULTS.items():
        if key not in out:
            continue
        if not hasattr(schema, key):
            continue
        if getattr(schema, key) == default:
            out.pop(key, None)

    if getattr(schema, "index", None) is None:
        out.pop("index", None)

    lib = out.get("dataframe_library")
    if lib in (None, "pandas"):
        out.pop("dataframe_library", None)

    df_checks = getattr(schema, "checks", None)
    if not df_checks:
        out.pop("checks", None)
    elif isinstance(out.get("checks"), list):
        _prune_check_list(out["checks"], list(df_checks))

    cols = out.get("columns")
    col_objs = getattr(schema, "columns", None)
    if isinstance(cols, dict) and col_objs:
        for name, ser in list(cols.items()):
            if not isinstance(ser, dict):
                continue
            if name not in col_objs:
                continue
            prune_component_dict(ser, col_objs[name])

    idx_ser = out.get("index")
    idx_parts = _index_components(schema)
    if isinstance(idx_ser, list) and idx_parts:
        for i, ser in enumerate(idx_ser):
            if i < len(idx_parts) and isinstance(ser, dict):
                prune_component_dict(ser, idx_parts[i])


# --- xarray DataArraySchema -------------------------------------------------
DATA_ARRAY_DEFAULTS: dict[str, Any] = {
    "ordered_dims": True,
    "coerce": False,
    "nullable": False,
    "name": None,
    "title": None,
    "description": None,
    "dims": None,
    "sizes": None,
    "shape": None,
}


def _prune_data_array_like(
    out: dict[str, Any], schema: Any, *, pop_version: bool
) -> None:
    if pop_version:
        out.pop("version", None)
    for key, default in DATA_ARRAY_DEFAULTS.items():
        if key not in out:
            continue
        if not hasattr(schema, key):
            continue
        cur = getattr(schema, key)
        if cur == default:
            out.pop(key, None)

    if not getattr(schema, "checks", None):
        out.pop("checks", None)
    elif isinstance(out.get("checks"), list):
        _prune_check_list(out["checks"], list(schema.checks))

    coords = out.get("coords")
    coord_schema = getattr(schema, "coords", None)
    if isinstance(coords, dict) and isinstance(coord_schema, dict):
        for cname, ser in list(coords.items()):
            if cname not in coord_schema or not isinstance(ser, dict):
                continue
            obj = coord_schema[cname]
            if obj is not None and hasattr(obj, "nullable"):
                prune_component_dict(ser, obj)


def apply_minimal_data_array(out: dict[str, Any], schema: Any) -> None:
    _prune_data_array_like(out, schema, pop_version=True)


def apply_minimal_nested_data_array(out: dict[str, Any], schema: Any) -> None:
    """Prune stats dict from :func:`get_data_array_schema_statistics` (no version)."""
    _prune_data_array_like(out, schema, pop_version=False)


# --- xarray DatasetSchema ---------------------------------------------------
DATASET_SCHEMA_DEFAULTS: dict[str, Any] = {
    "ordered_dims": True,
    "strict": False,
    "strict_coords": False,
    "strict_attrs": False,
    "name": None,
    "title": None,
    "description": None,
}


def apply_minimal_dataset(out: dict[str, Any], schema: Any) -> None:
    from pandera.api.xarray.container import DataArraySchema

    out.pop("version", None)
    for key, default in DATASET_SCHEMA_DEFAULTS.items():
        if key not in out:
            continue
        if hasattr(schema, key) and getattr(schema, key) == default:
            out.pop(key, None)

    if not getattr(schema, "checks", None):
        out.pop("checks", None)
    elif isinstance(out.get("checks"), list):
        _prune_check_list(out["checks"], list(schema.checks))

    dvs = out.get("data_vars")
    dv_schema = getattr(schema, "data_vars", None)
    if isinstance(dvs, dict) and isinstance(dv_schema, dict):
        for key, ser in list(dvs.items()):
            if key not in dv_schema or not isinstance(ser, dict):
                continue
            spec = dv_schema[key]
            if spec is None:
                continue
            if isinstance(spec, DataArraySchema):
                apply_minimal_nested_data_array(ser, spec)
            elif hasattr(spec, "dtype"):
                prune_component_dict(ser, spec)

    coords = out.get("coords")
    coord_schema = getattr(schema, "coords", None)
    if isinstance(coords, dict) and isinstance(coord_schema, dict):
        for cname, ser in list(coords.items()):
            if cname not in coord_schema or not isinstance(ser, dict):
                continue
            obj = coord_schema[cname]
            if obj is not None and hasattr(obj, "nullable"):
                prune_component_dict(ser, obj)
