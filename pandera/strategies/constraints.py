"""Aggregated, dtype-agnostic field constraints for data synthesis.

This module defines :class:`FieldConstraints`, a value type that captures
the *intersection* of all numeric/string/membership constraints derived
from a list of :class:`~pandera.api.checks.Check` instances on a single
field (a pandas Series, Index, Column, or xarray DataArray/data variable).

The constraint aggregator is the foundation of the optimised
hypothesis-based data-synthesis layer: rather than chaining one
``.filter`` per check on the resulting strategy, sibling constraints are
merged ahead of time and a single hypothesis strategy is constructed
from the merged result.

See :doc:`/data_synthesis_strategies` for user-facing guidance and the
``specs/optimized-strategies.md`` design doc for the rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any


class _UnsetType:
    """Sentinel type for unset constraint fields.

    Distinct from ``None`` because some constraint values legitimately
    accept ``None`` as a meaningful value (e.g. coordinate of NaN).
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Any = _UnsetType()


Predicate = Callable[[Any], bool]
PostMergeHook = Callable[["FieldConstraints"], "FieldConstraints"]


class ConstraintConflictError(ValueError):
    """Raised when two ``FieldConstraints`` cannot be jointly satisfied.

    The constraint aggregator raises this exception during
    :meth:`FieldConstraints.merge` (or during
    :func:`~pandera.strategies.pandas_strategies.compile_field_strategy`)
    when the merged constraint set has an empty satisfying region:

    - numeric inversion (``min_value > max_value``),
    - empty ``isin`` set,
    - ``eq`` value that is also in ``notin``,
    - ``isin`` set that does not contain ``eq``,
    - mismatched ``eq`` values,
    - mismatched ``str_exact_len`` values,
    - bounds that exclude all ``isin`` candidates.

    Callers in the strategy layer translate this into a
    ``SchemaDefinitionError`` so the user gets an actionable error at
    strategy-construction time rather than after ``hypothesis`` exhausts
    itself.
    """


@dataclass(frozen=True)
class FieldConstraints:
    """Aggregated, dtype-agnostic constraints for a single field.

    All attributes default to ``UNSET`` (numeric bounds, ``eq``) or a
    "no constraint" value (``None`` for ``isin``, empty for ``notin`` /
    ``regex_*`` / ``residual_filters`` / ``post_merge_hooks``).

    :meth:`merge` intersects two ``FieldConstraints`` (tightest wins) and
    raises :class:`ConstraintConflictError` if they are jointly
    unsatisfiable.
    """

    # Numeric / orderable
    min_value: Any = UNSET
    max_value: Any = UNSET
    exclude_min: bool = False
    exclude_max: bool = False

    # Membership
    isin: frozenset | None = None
    notin: frozenset = field(default_factory=frozenset)

    # Equality (collapses to a single value)
    eq: Any = UNSET

    # String regex / length
    regex_fullmatch: tuple[str, ...] = ()
    regex_search: tuple[str, ...] = ()
    str_min_len: int | None = None
    str_max_len: int | None = None
    str_exact_len: int | None = None

    # Floats / complex
    allow_nan: bool = False
    allow_infinity: bool = False

    # Catch-all opaque per-check predicates that we could not aggregate.
    # Applied as a single trailing ``.filter(...)`` chain at the end of
    # ``compile_field_strategy``.
    residual_filters: tuple[tuple[str, Predicate], ...] = ()

    # Optional adapter-supplied callbacks of type
    # ``(FieldConstraints) -> FieldConstraints`` run *after* the final
    # ``merge`` and *before* ``compile_field_strategy``. They let an
    # adapter rewrite itself with knowledge of what sibling checks
    # contributed (e.g. upgrading a residual filter to an ``isin`` set
    # when bounds are known).
    post_merge_hooks: tuple[PostMergeHook, ...] = ()

    def merge(self, other: FieldConstraints) -> FieldConstraints:
        """Intersect this ``FieldConstraints`` with another.

        :param other: another ``FieldConstraints`` whose constraints
            will be combined with this one (tightest wins).
        :returns: a new ``FieldConstraints``.
        :raises ConstraintConflictError: if the merged constraints are
            jointly unsatisfiable.
        """
        merged = replace(
            self,
            min_value=_merge_min(self, other),
            max_value=_merge_max(self, other),
            exclude_min=_merge_exclude_min(self, other),
            exclude_max=_merge_exclude_max(self, other),
            isin=_merge_isin(self, other),
            notin=self.notin | other.notin,
            eq=_merge_eq(self, other),
            regex_fullmatch=self.regex_fullmatch + other.regex_fullmatch,
            regex_search=self.regex_search + other.regex_search,
            str_min_len=_merge_max_int(self.str_min_len, other.str_min_len),
            str_max_len=_merge_min_int(self.str_max_len, other.str_max_len),
            str_exact_len=_merge_str_exact_len(
                self.str_exact_len, other.str_exact_len
            ),
            allow_nan=self.allow_nan and other.allow_nan,
            allow_infinity=self.allow_infinity and other.allow_infinity,
            residual_filters=self.residual_filters + other.residual_filters,
            post_merge_hooks=self.post_merge_hooks + other.post_merge_hooks,
        )
        merged.validate()
        return merged

    def validate(self) -> None:
        """Verify the constraint set is internally consistent.

        :raises ConstraintConflictError: when the constraints describe
            an empty satisfying region.
        """
        if (
            self.min_value is not UNSET
            and self.max_value is not UNSET
            and self.min_value > self.max_value
        ):
            raise ConstraintConflictError(
                f"min_value={self.min_value!r} > max_value={self.max_value!r}"
            )

        if (
            self.min_value is not UNSET
            and self.max_value is not UNSET
            and self.min_value == self.max_value
            and (self.exclude_min or self.exclude_max)
        ):
            raise ConstraintConflictError(
                f"min_value={self.min_value!r} == "
                f"max_value={self.max_value!r} with exclusive bounds "
                "describes an empty interval"
            )

        if self.isin is not None and len(self.isin) == 0:
            raise ConstraintConflictError("isin constraint is an empty set")

        if self.eq is not UNSET:
            if self.eq in self.notin:
                raise ConstraintConflictError(
                    f"eq={self.eq!r} conflicts with notin={self.notin!r}"
                )
            if self.isin is not None and self.eq not in self.isin:
                raise ConstraintConflictError(
                    f"eq={self.eq!r} not in isin={self.isin!r}"
                )
            if self.min_value is not UNSET and (
                self.eq < self.min_value
                or (self.eq == self.min_value and self.exclude_min)
            ):
                raise ConstraintConflictError(
                    f"eq={self.eq!r} violates min_value={self.min_value!r}"
                )
            if self.max_value is not UNSET and (
                self.eq > self.max_value
                or (self.eq == self.max_value and self.exclude_max)
            ):
                raise ConstraintConflictError(
                    f"eq={self.eq!r} violates max_value={self.max_value!r}"
                )

        if (
            self.str_exact_len is not None
            and self.str_min_len is not None
            and self.str_min_len > self.str_exact_len
        ):
            raise ConstraintConflictError(
                f"str_exact_len={self.str_exact_len} < "
                f"str_min_len={self.str_min_len}"
            )
        if (
            self.str_exact_len is not None
            and self.str_max_len is not None
            and self.str_max_len < self.str_exact_len
        ):
            raise ConstraintConflictError(
                f"str_exact_len={self.str_exact_len} > "
                f"str_max_len={self.str_max_len}"
            )
        if (
            self.str_min_len is not None
            and self.str_max_len is not None
            and self.str_min_len > self.str_max_len
        ):
            raise ConstraintConflictError(
                f"str_min_len={self.str_min_len} > "
                f"str_max_len={self.str_max_len}"
            )

    def is_empty(self) -> bool:
        """Return ``True`` when no constraints have been set."""
        return self == FieldConstraints()

    def apply_post_merge_hooks(self) -> FieldConstraints:
        """Apply registered ``post_merge_hooks`` left-to-right.

        Each hook receives the (potentially already rewritten)
        ``FieldConstraints`` and returns a new one. Hooks are stripped
        from the result so they don't run twice.
        """
        result = self
        if not result.post_merge_hooks:
            return result
        hooks = result.post_merge_hooks
        result = replace(result, post_merge_hooks=())
        for hook in hooks:
            result = hook(result)
        result.validate()
        return result


def _merge_min(left: FieldConstraints, right: FieldConstraints) -> Any:
    if left.min_value is UNSET:
        return right.min_value
    if right.min_value is UNSET:
        return left.min_value
    return max(left.min_value, right.min_value)


def _merge_max(left: FieldConstraints, right: FieldConstraints) -> Any:
    if left.max_value is UNSET:
        return right.max_value
    if right.max_value is UNSET:
        return left.max_value
    return min(left.max_value, right.max_value)


def _merge_exclude_min(
    left: FieldConstraints, right: FieldConstraints
) -> bool:
    if left.min_value is UNSET:
        return right.exclude_min
    if right.min_value is UNSET:
        return left.exclude_min
    if left.min_value == right.min_value:
        return left.exclude_min or right.exclude_min
    return (
        left.exclude_min
        if left.min_value > right.min_value
        else (right.exclude_min)
    )


def _merge_exclude_max(
    left: FieldConstraints, right: FieldConstraints
) -> bool:
    if left.max_value is UNSET:
        return right.exclude_max
    if right.max_value is UNSET:
        return left.exclude_max
    if left.max_value == right.max_value:
        return left.exclude_max or right.exclude_max
    return (
        left.exclude_max
        if left.max_value < right.max_value
        else (right.exclude_max)
    )


def _merge_isin(
    left: FieldConstraints, right: FieldConstraints
) -> frozenset | None:
    if left.isin is None:
        return right.isin
    if right.isin is None:
        return left.isin
    return left.isin & right.isin


def _merge_eq(left: FieldConstraints, right: FieldConstraints) -> Any:
    if left.eq is UNSET:
        return right.eq
    if right.eq is UNSET:
        return left.eq
    if left.eq != right.eq:
        raise ConstraintConflictError(
            f"conflicting eq values: {left.eq!r} != {right.eq!r}"
        )
    return left.eq


def _merge_min_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _merge_max_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _merge_str_exact_len(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    if left != right:
        raise ConstraintConflictError(
            f"conflicting str_exact_len values: {left} != {right}"
        )
    return left


__all__ = [
    "FieldConstraints",
    "ConstraintConflictError",
    "Predicate",
    "PostMergeHook",
    "UNSET",
]
