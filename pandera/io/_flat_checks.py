"""Flatten column/index checks for YAML/JSON (schema-style keys).

Serializes checks as ``{check_name: value}`` instead of a ``checks:`` list, matching
the ergonomics of :func:`~pandera.api.dataframe.model_components.Field`.
Deserialization accepts both shapes.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from pandera.api.checks import Check
from pandera.io._minimal import CHECK_OPTION_DEFAULTS

# Keys that may appear on serialized column/index/component dicts — not check names.
COMPONENT_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "title",
        "description",
        "dtype",
        "nullable",
        "checks",
        "name",
        "unique",
        "coerce",
        "required",
        "regex",
        "default",
        "report_duplicates",
        "drop_invalid_rows",
        "dims",
        "alias",
        "ordered_dims",
        "sizes",
        "shape",
        "coords",
        "schema_type",
        "version",
    }
)

CHECK_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "check_name",
        "ignore_na",
        "raise_warning",
        "n_failure_cases",
    }
)


def _prune_default_options(opt_rest: dict[str, Any]) -> None:
    for ok, ov in list(opt_rest.items()):
        if ok in CHECK_OPTION_DEFAULTS and ov == CHECK_OPTION_DEFAULTS[ok]:
            del opt_rest[ok]


def flatten_check_list_entry(
    entry: Any,
) -> tuple[str | None, Any]:
    """Return ``(check_name, flat_value)`` or ``(None, None)`` if not flattenable."""
    if not isinstance(entry, dict):
        return None, None
    options = entry.get("options") or {}
    if not isinstance(options, dict):
        return None, None
    check_name = options.get("check_name")
    if not check_name:
        return None, None
    stats = {k: v for k, v in entry.items() if k != "options"}
    opt_rest = {k: v for k, v in options.items() if k != "check_name"}
    _prune_default_options(opt_rest)

    if not stats and not opt_rest:
        return None, None

    if len(stats) == 1 and not opt_rest:
        return check_name, next(iter(stats.values()))

    if len(stats) == 1 and opt_rest:
        key = next(iter(stats.keys()))
        return check_name, {key: stats[key], **opt_rest}

    if opt_rest:
        return check_name, {**stats, **opt_rest}
    return check_name, stats


def flatten_component_checks_dict(d: MutableMapping[str, Any]) -> None:
    """Replace ``checks: [ ... ]`` with ``check_name: value`` keys (in-place)."""
    chk = d.get("checks")
    if chk is None:
        return
    if isinstance(chk, dict):
        return
    if not isinstance(chk, list):
        d.pop("checks", None)
        return
    if not chk:
        d.pop("checks", None)
        return

    flat: dict[str, Any] = {}
    for item in chk:
        name, value = flatten_check_list_entry(item)
        if name is None:
            return
        if name in flat:
            return
        flat[name] = value

    d.pop("checks", None)
    d.update(flat)


def flat_value_to_list_entry(check_name: str, val: Any) -> dict[str, Any]:
    """Turn a schema-style ``check_name: value`` into list-format check stats."""
    if val is None:
        return {"options": {"check_name": check_name}}
    if not isinstance(val, dict):
        return {"value": val, "options": {"check_name": check_name}}

    opts: dict[str, Any] = {"check_name": check_name}
    stats: dict[str, Any] = {}
    for kk, vv in val.items():
        if kk in CHECK_OPTION_KEYS and kk != "check_name":
            opts[kk] = vv
        else:
            stats[kk] = vv
    if not stats:
        return {"options": opts}
    return {**stats, "options": opts}


def unflatten_component_checks_dict(d: MutableMapping[str, Any]) -> None:
    """Expand flat check keys into ``checks: [ ... ]`` when needed (in-place)."""
    if d.get("checks") is not None:
        return

    check_keys: list[str] = []
    for k in d:
        if k in COMPONENT_RESERVED_KEYS:
            continue
        fn = getattr(Check, k, None)
        if fn is None or not callable(fn):
            continue
        check_keys.append(k)

    if not check_keys:
        return

    checks: list[dict[str, Any]] = []
    for k in check_keys:
        val = d.pop(k)
        checks.append(flat_value_to_list_entry(k, val))
    d["checks"] = checks


def apply_flat_checks_to_dataframe_serialized(out: dict[str, Any]) -> None:
    """Flatten column/index check lists on a serialized dataframe schema dict."""
    cols = out.get("columns")
    if isinstance(cols, dict):
        for ser in cols.values():
            if isinstance(ser, dict):
                flatten_component_checks_dict(ser)
    idx = out.get("index")
    if isinstance(idx, list):
        for ser in idx:
            if isinstance(ser, dict):
                flatten_component_checks_dict(ser)


def apply_flat_checks_to_xarray_serialized(out: dict[str, Any]) -> None:
    """Flatten checks on coordinate and data var components (not schema-level)."""
    coords = out.get("coords")
    if isinstance(coords, dict):
        for ser in coords.values():
            if isinstance(ser, dict):
                flatten_component_checks_dict(ser)

    dvs = out.get("data_vars")
    if isinstance(dvs, dict):
        for ser in dvs.values():
            if isinstance(ser, dict):
                flatten_component_checks_dict(ser)
                nested_coords = ser.get("coords")
                if isinstance(nested_coords, dict):
                    for cser in nested_coords.values():
                        if isinstance(cser, dict):
                            flatten_component_checks_dict(cser)
