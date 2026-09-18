"""Shared, dataframe-library-agnostic statistics helpers.

This module must stay free of pandas (and other dataframe-library) imports so
that pandas-free entrypoints like ``pandera.polars`` can use it.
"""

from __future__ import annotations

import warnings
from typing import Any, Union

from pandera.api.checks import Check

# Group keys survive YAML/JSON only as these exact types. PyYAML looks up a
# representer by the exact type, so a numpy scalar raises even when it
# subclasses a builtin (``np.float64`` is a ``float``).
_SERIALIZABLE_KEY_TYPES = (str, bool, int, float, type(None))


class UnserializableGroupKey(Exception):
    """A group key has no faithful YAML/JSON-serializable equivalent."""


def _serialize_group_key(key: Any, *, allow_tuples: bool) -> Any:
    """Convert one group key into a serializable equivalent.

    The equivalent has to keep selecting the same group once the schema is
    read back, so only conversions that compare and hash equal to the original
    are allowed. Anything else raises :class:`UnserializableGroupKey`.
    """
    if type(key) in _SERIALIZABLE_KEY_TYPES:
        return key

    # A tuple key (multi-column groupby) serializes as a list and is restored
    # by `deserialize_group_keys`, since a group key is always hashable. A
    # column name has no such marker, so tuples are not allowed there.
    if allow_tuples and isinstance(key, tuple):
        return [_serialize_group_key(k, allow_tuples=True) for k in key]

    # Numpy scalars carry `.item()`, which yields the equivalent builtin.
    item = getattr(key, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError) as exc:
            raise UnserializableGroupKey(key) from exc
        if type(value) in _SERIALIZABLE_KEY_TYPES:
            return value

    raise UnserializableGroupKey(key)


def serialize_group_keys(groups, *, allow_tuples: bool = True):
    """Convert ``Check.groups`` or ``Check.groupby`` into serializable values.

    :param allow_tuples: whether a tuple is a valid value. True for group
        keys, False for the column names in ``groupby``.
    :raises UnserializableGroupKey: if a value has no faithful equivalent.
    """
    if groups is None:
        return None
    if isinstance(groups, (list, tuple)):
        return [
            _serialize_group_key(key, allow_tuples=allow_tuples)
            for key in groups
        ]
    return _serialize_group_key(groups, allow_tuples=allow_tuples)


def deserialize_group_keys(groups):
    """Restore ``Check.groups`` read from YAML/JSON.

    A group key is hashable by definition, so a list can only be a tuple key
    that lost its type on the way out.
    """
    if isinstance(groups, list):
        return [
            tuple(deserialize_group_keys(key))
            if isinstance(key, list)
            else key
            for key in groups
        ]
    return groups


def string_length_check_statistics(
    min_len: int, max_len: int
) -> dict[str, Any]:
    """Build ``parse_check_statistics``-compatible stats for
    :meth:`~pandera.api.checks.Check.str_length`.
    """
    return {
        "str_length": {
            "min_value": min_len,
            "max_value": max_len,
        },
    }


def parse_check_statistics(check_stats: Union[dict[str, Any], None]):
    """Convert check statistics to a list of Check objects, including their options."""
    if check_stats is None:
        return None
    checks = []
    for check_name, stats in check_stats.items():
        check = getattr(Check, check_name)
        try:
            # Extract options if present
            if isinstance(stats, dict):
                options = (
                    stats.pop("options", {}) if "options" in stats else {}
                )
                if stats:  # If there are remaining stats
                    check_instance = check(**stats)
                else:  # Handle case where all stats were in options
                    check_instance = check()
                # Apply options to the check instance
                for option_name, option_value in options.items():
                    if option_name == "groups":
                        option_value = deserialize_group_keys(option_value)
                    setattr(check_instance, option_name, option_value)
                checks.append(check_instance)
            else:
                # Handle unary check case
                checks.append(check(stats))
        except TypeError:
            # if stats cannot be unpacked as key-word args, assume unary check.
            checks.append(check(stats))
    return checks if checks else None


def parse_checks(checks) -> Union[list[dict[str, Any]], None]:
    """Convert Check object to check statistics including options."""

    def _has_custom_error(check: Check, registration_name: str) -> bool:
        """Determine whether a check has a user-defined error message."""
        if check.error is None:
            return False

        if not Check.is_builtin_check(registration_name):
            return True

        try:
            default_check = getattr(Check, registration_name)(
                **(check.statistics or {})
            )
        except (AttributeError, TypeError, ValueError):
            return True

        return check.error != default_check.error

    check_statistics = []

    for check in checks:
        registration_name = check.registry_name
        if registration_name is None:
            warnings.warn(
                "Only registered checks may be serialized to statistics. "
                "Did you forget to register it with the extension API? "
                f"Check `{check.name}` will be skipped."
            )
            continue

        if callable(check.groupby):
            warnings.warn(
                "Checks with a callable `groupby` cannot be serialized to "
                f"statistics. Check `{check.name}` will be skipped."
            )
            continue

        try:
            groupby = serialize_group_keys(check.groupby, allow_tuples=False)
            groups = serialize_group_keys(check.groups)
        except UnserializableGroupKey as exc:
            warnings.warn(
                "Checks with a `groupby` or `groups` value that has no "
                f"serializable equivalent ({exc.args[0]!r}) cannot be "
                f"serialized to statistics. Check `{check.name}` will be "
                "skipped."
            )
            continue

        # Get base statistics
        base_stats = {} if check.statistics is None else check.statistics

        # Collect check options
        check_options = {
            "check_name": registration_name,
            "raise_warning": check.raise_warning,
            "n_failure_cases": check.n_failure_cases,
            "ignore_na": check.ignore_na,
            "groupby": groupby,
            "groups": groups,
        }
        if check.name != registration_name:
            check_options["name"] = check.name
        if _has_custom_error(check, registration_name):
            check_options["error"] = check.error

        # Filter out None values from options
        check_options = {
            k: v for k, v in check_options.items() if v is not None
        }

        # Combine statistics with options
        if check_options:
            base_stats["options"] = check_options
            check_statistics.append(base_stats)

    return check_statistics if check_statistics else None
