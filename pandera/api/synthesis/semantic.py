"""Semantic typing for realistic data synthesis.

A generator cannot infer from a dtype alone whether a ``str`` column is a
*name*, an *email*, or a *zip code* — all are ``str``. Realism therefore needs
an explicit, opt-in ``semantic`` signal. :class:`SemanticType` enumerates the
semantic categories the built-in backends understand.

The ``semantic`` tag is *pure metadata* for synthesis. It has **no validation
semantics** and never affects ``validate()``. If a ``semantic`` is
untranslatable to a realistic provider, synthesis silently falls back to the
default (pure-``hypothesis``) path.
"""

from __future__ import annotations

from enum import Enum


class SemanticType(str, Enum):
    """Semantic category of a field, used to drive realistic generation.

    Attached to a :class:`~pandera.api.pandas.components.Column`, Index, or
    :func:`~pandera.api.dataframe.model_components.Field` via the ``semantic=``
    keyword. A backend maps each value to a realistic provider; unmapped or
    infeasible semantics cause a graceful fallback to hypothesis.
    """

    NAME = "name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CITY = "city"
    STATE = "state"
    POSTAL_CODE = "postal_code"
    COUNTRY = "country"
    COMPANY = "company"
    JOB = "job"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CURRENCY = "currency"
    UUID = "uuid"


def coerce_semantic(
    semantic: SemanticType | str | None,
) -> SemanticType | None:
    """Coerce a ``semantic`` value into a :class:`SemanticType` or ``None``.

    ``None`` passes through unchanged; strings are validated against the enum
    (raising ``ValueError`` for unknown values so schema authors fail fast at
    definition time).

    :param semantic: a :class:`SemanticType`, its string value, or ``None``.
    :returns: the canonical :class:`SemanticType`, or ``None``.
    :raises ValueError: if ``semantic`` is an unrecognised string.
    """
    if semantic is None:
        return None
    if isinstance(semantic, SemanticType):
        return semantic
    return SemanticType(semantic)


__all__ = ["SemanticType", "coerce_semantic"]
