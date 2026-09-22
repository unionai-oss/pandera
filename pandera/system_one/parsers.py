"""The three System One column parsers: ``Choice``, ``Score`` and ``Noul``.

Each is a :class:`~pandera.api.parsers.ColumnParser`, so asking a decision
model to fill a column is declared exactly like any other derived column. What
is asked comes from the column itself -- its declared type supplies the
options, its ``description`` supplies the question, and enum member docstrings
supply the criteria -- so nothing has to be repeated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd

from pandera import _enum_literal
from pandera.api.parsers import ParseContext
from pandera.errors import SchemaInitError
from pandera.system_one import questions as q
from pandera.system_one.execution import gather_decisions, run_sync
from pandera.system_one.providers import base as provider_base

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class _SystemOneParser:
    """Shared machinery for the three question types."""

    # These parsers fill several columns from one request, so they always
    # receive the source columns as a DataFrame.
    frame_input = True

    def __init__(
        self,
        instructions: str | None = None,
        criteria: Any = None,
        *,
        provider: Any = None,
    ):
        self.instructions = instructions
        self.criteria = criteria
        self.provider = provider

    # -- protocol ---------------------------------------------------------

    def batch_key(self, ctx: ParseContext) -> Any:
        """Group by who answers and what they are shown.

        Columns sharing a provider, a source and an error policy can be filled
        by one request, because a System One request is one state plus many
        questions. This is what keeps a per-column declaration from costing a
        per-column request.
        """
        provider_id = (
            self.provider
            if isinstance(self.provider, str)
            else getattr(self.provider, "id", None)
        )
        return (provider_id, ctx.source, ctx.on_error)

    def bind(self, ctx: ParseContext):
        """Satisfy the protocol for a group of one."""
        return type(self).batch([(self, ctx)])

    @classmethod
    def batch(cls, items: Sequence[tuple[Any, ParseContext]]):
        """Compile a group of columns into a single request per row."""
        compiled = [
            (parser, ctx, parser.to_question(ctx)) for parser, ctx in items
        ]
        questions = {ctx.target: question for _, ctx, question in compiled}
        targets = [ctx.target for _, ctx, _ in compiled]
        first_ctx = compiled[0][1]
        declared_provider = next(
            (parser.provider for parser, _, _ in compiled if parser.provider),
            None,
        )
        on_error = first_ctx.on_error

        def _fn(df: pd.DataFrame) -> pd.DataFrame:
            provider = (
                provider_base._coerce(declared_provider)
                if declared_provider is not None
                else provider_base.require_provider(targets)
            )
            states = _build_states(df, first_ctx.source)
            prepared = provider.compile(questions)

            async def _decide(state: Any) -> Mapping[str, q.Decision]:
                return await provider.decide(state, prepared)

            answers = run_sync(
                gather_decisions(
                    states,
                    _decide,
                    provider.limits,
                    on_error=on_error,
                )
            )

            out = df.copy()
            for parser, ctx, question in compiled:
                values = [
                    parser.to_value(question, ctx, row and row.get(ctx.target))
                    for row in answers
                ]
                out[ctx.target] = _as_declared(values, df.index, ctx.dtype)
            return out

        return _fn

    # -- subclass hooks ---------------------------------------------------

    def to_question(self, ctx: ParseContext) -> q.Question:
        raise NotImplementedError  # pragma: no cover

    def to_value(
        self, question: q.Question, ctx: ParseContext, decision: Any
    ) -> Any:
        raise NotImplementedError  # pragma: no cover

    # -- helpers ----------------------------------------------------------

    def _instructions(self, ctx: ParseContext) -> str:
        if self.instructions is not None:
            return self.instructions
        if ctx.description:
            return ctx.description
        raise SchemaInitError(
            f"column '{ctx.target}' is filled by "
            f"{type(self).__name__}() but has no question to ask. Give the "
            "field a `description`, or pass `instructions=` to the parser."
        )


class Noul(_SystemOneParser):
    """A yes/no question, answered with a calibrated probability.

    Fills a ``bool`` column by thresholding the probability, or a ``float``
    column with the probability itself.
    """

    def __init__(
        self,
        instructions: str | None = None,
        *,
        threshold: float = 0.5,
        true_description: str | None = None,
        false_description: str | None = None,
        provider: Any = None,
    ):
        super().__init__(instructions, provider=provider)
        self.threshold = threshold
        self.true_description = true_description
        self.false_description = false_description

    def to_question(self, ctx: ParseContext) -> q.Noul:
        kind = _dtype_kind(ctx.dtype)
        if kind not in ("bool", "float"):
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Noul(), which answers a "
                f"probability, so its type must be `bool` (thresholded) or "
                f"`float` (the raw probability). Got {ctx.dtype!r}."
            )
        return q.Noul(
            instructions=self._instructions(ctx),
            true_description=self.true_description,
            false_description=self.false_description,
        )

    def to_value(
        self, question: q.Question, ctx: ParseContext, decision: Any
    ) -> Any:
        if decision is None:
            return None
        probability = float(decision.value)
        # A bool column wants a decision, a float column wants the calibrated
        # probability itself.
        if _dtype_kind(ctx.dtype) == "bool":
            return probability >= self.threshold
        return probability


class Choice(_SystemOneParser):
    """A question that selects one of the column's options.

    Options come from the column's declared type -- an ``Enum``, a
    ``Literal``, or a categorical -- and their descriptions from enum member
    docstrings, so the taxonomy is declared once.
    """

    def to_question(self, ctx: ParseContext) -> q.Choice:
        options = _categories(ctx.dtype)
        if options is None:
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Choice(), which selects "
                "one of a fixed set of options, so its type must be an Enum, "
                f"a Literal, or a categorical. Got {ctx.dtype!r}."
            )
        if len(options) < 2:
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Choice() but its type "
                f"offers {len(options)} option(s). At least 2 are needed."
            )
        if len(options) > MAX_CHOICE_OPTIONS:
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Choice() but its type "
                f"offers {len(options)} options, more than the "
                f"{MAX_CHOICE_OPTIONS} a System One choice supports."
            )

        descriptions = _descriptions(ctx.dtype, options)
        if self.criteria is not None:
            if not isinstance(self.criteria, Mapping):
                raise SchemaInitError(
                    f"column '{ctx.target}': Choice(criteria=...) must be a "
                    f"mapping of option to description, got "
                    f"{type(self.criteria).__name__}."
                )
            unknown = set(self.criteria) - {str(option) for option in options}
            if unknown:
                raise SchemaInitError(
                    f"column '{ctx.target}': Choice(criteria=...) mentions "
                    f"{sorted(unknown)}, which {'is' if len(unknown) == 1 else 'are'} "
                    f"not among the column's options {[str(o) for o in options]}."
                )
            descriptions = {
                option: self.criteria.get(
                    str(option), descriptions.get(option)
                )
                for option in options
            }

        return q.Choice(
            instructions=self._instructions(ctx),
            criteria=tuple(
                (str(option), descriptions.get(option)) for option in options
            ),
            values=tuple(options),
            allow_none=ctx.nullable,
        )

    def to_value(
        self, question: q.Question, ctx: ParseContext, decision: Any
    ) -> Any:
        if decision is None:
            return None
        # Answers come back as option labels; map them to the column's own
        # value domain, which may not be strings.
        return _label_to_value(cast(q.Choice, question), decision.value)


class Score(_SystemOneParser):
    """A question that places the state on the column's ordered scale.

    Levels come from an ordered type -- typically an ``IntEnum`` -- in value
    order, and the rubric from its member docstrings. Naming ``Score()``
    rather than inferring it from ordering is deliberate: it is one word, and
    an unordered type under ``Score()`` is an error rather than a silently
    different question.
    """

    def to_question(self, ctx: ParseContext) -> q.Score:
        levels = _categories(ctx.dtype)
        if levels is None:
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Score(), which rates "
                "against an ordered rubric, so its type must be an ordered "
                f"categorical such as an IntEnum. Got {ctx.dtype!r}."
            )
        if not _is_ordered(ctx.dtype):
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Score() but its type is "
                "unordered, so there is no scale to rate against. Use an "
                "IntEnum, pass dtype_kwargs={'ordered': True}, or use "
                "Choice() if the options are not ranked."
            )
        if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
            raise SchemaInitError(
                f"column '{ctx.target}' is filled by Score() but its type has "
                f"{len(levels)} levels. A System One score supports "
                f"{MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS}."
            )

        descriptions = _descriptions(ctx.dtype, levels)
        if self.criteria is not None:
            if isinstance(self.criteria, Mapping) or not isinstance(
                self.criteria, Sequence
            ):
                raise SchemaInitError(
                    f"column '{ctx.target}': Score(criteria=...) must be an "
                    "ordered sequence of level descriptions, one per level."
                )
            if len(self.criteria) != len(levels):
                raise SchemaInitError(
                    f"column '{ctx.target}': Score(criteria=...) has "
                    f"{len(self.criteria)} descriptions but the column has "
                    f"{len(levels)} levels."
                )
            rubric = tuple(str(item) for item in self.criteria)
        else:
            missing = [
                level for level in levels if not descriptions.get(level)
            ]
            if missing:
                # Bare level names make a poor rubric, and the model has
                # nothing else to go on, so say so rather than degrade.
                rubric = tuple(str(level) for level in levels)
            else:
                rubric = tuple(str(descriptions[level]) for level in levels)

        return q.Score(
            instructions=self._instructions(ctx),
            criteria=rubric,
            levels=tuple(levels),
        )

    def to_value(
        self, question: q.Question, ctx: ParseContext, decision: Any
    ) -> Any:
        if decision is None:
            return None
        score = float(decision.value)
        # A score may land between levels. A float column keeps the unrounded
        # position; anything else snaps to the nearest declared level.
        if _dtype_kind(ctx.dtype) == "float":
            return score
        levels = cast(q.Score, question).levels
        index = max(0, min(int(round(score)), len(levels) - 1))
        return levels[index]


# ---------------------------------------------------------------------------
# type inspection
# ---------------------------------------------------------------------------


def _dtype_kind(dtype: Any) -> str:
    """Coarse classification of a column's declared type."""
    if dtype is None:
        return "unknown"
    text = str(dtype).lower()
    if text.startswith("bool"):
        return "bool"
    if text.startswith("float"):
        return "float"
    if text in ("category", "categorical") or hasattr(dtype, "categories"):
        return "category"
    if text.startswith(("int", "uint")):
        return "int"
    return text


def _categories(dtype: Any) -> tuple[Any, ...] | None:
    """The option set a categorical column declares, if it has one."""
    categories = getattr(dtype, "categories", None)
    if categories is None:
        inner = getattr(dtype, "type", None)
        categories = getattr(inner, "categories", None)
    if categories is None:
        return None
    return tuple(categories)


def _is_ordered(dtype: Any) -> bool:
    ordered = getattr(dtype, "ordered", None)
    if ordered is None:
        ordered = getattr(getattr(dtype, "type", None), "ordered", None)
    return bool(ordered)


def _descriptions(
    dtype: Any, categories: Sequence[Any]
) -> dict[Any, str | None]:
    """Per-option descriptions, from enum member docstrings when available."""
    enum_type = getattr(dtype, "enum_type", None)
    if enum_type is not None:
        return _enum_literal.value_descriptions(enum_type)
    return {category: None for category in categories}


def _label_to_value(question: q.Choice, label: Any) -> Any:
    """Map an answer label back to the column's own value domain.

    Labels are always strings on the wire, but the column's options may not
    be -- a categorical of ints, for instance.
    """
    if label is None:
        return None
    for (option_label, _), value in zip(question.criteria, question.values):
        if option_label == str(label):
            return value
    return label


def _build_states(
    df: pd.DataFrame, source: tuple[str, ...] | None
) -> list[Any]:
    """Build one state per row from the declared source columns.

    A single source column is passed as its value; several are passed as a
    mapping. Accuracy degrades with irrelevant context, so the whole row is
    never sent.
    """
    if source is None:
        return [row.to_dict() for _, row in df.iterrows()]
    if len(source) == 1:
        column = source[0]
        return [_jsonable(value) for value in df[column].tolist()]
    return [
        {column: _jsonable(row[column]) for column in source}
        for _, row in df[list(source)].iterrows()
    ]


def _as_declared(values: Sequence[Any], index: Any, dtype: Any) -> pd.Series:
    """Build the answer column in the type the schema declared.

    A System One answer is always within the column's domain by construction,
    so the parser produces the declared type directly rather than leaving an
    object column for coercion to clean up.
    """
    series = pd.Series(values, index=index, dtype="object")
    target = getattr(dtype, "type", None)
    if target is None:
        return series
    try:
        return series.astype(target)
    except (TypeError, ValueError):
        # Leave it alone and let the column's own dtype check report it --
        # a silently mistyped answer column would be worse than a clear error.
        return series


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
