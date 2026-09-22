"""Inspecting what a schema will ask, without asking it.

Filling a column by asking a model costs money and time, so it has to be
possible to see exactly what would be sent -- and what it would cost -- without
sending it. Neither function here needs a provider.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pandera.system_one.parsers import (
    Confidence,
    _build_states,
    _SystemOneParser,
)
from pandera.system_one.primitives import Question


@dataclasses.dataclass(frozen=True)
class Plan:
    """What a validation would send, and roughly what it would cost."""

    rows: int
    batches: int
    requests: int
    questions: int
    estimated_input_tokens: int
    estimated_cost_usd: float | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        cost = (
            "unknown"
            if self.estimated_cost_usd is None
            else f"${self.estimated_cost_usd:.4f}"
        )
        return (
            f"Plan(rows={self.rows}, batches={self.batches}, "
            f"requests={self.requests}, questions={self.questions}, "
            f"est_input_tokens={self.estimated_input_tokens}, "
            f"est_cost={cost})"
        )


def _as_schema(target: Any) -> Any:
    to_schema = getattr(target, "to_schema", None)
    return to_schema() if callable(to_schema) else target


def _groups(schema: Any) -> list[list[tuple[Any, Any]]]:
    """The batches a schema's System One columns would form."""
    batches: dict[Any, list[tuple[Any, Any]]] = {}
    for name, column in getattr(schema, "columns", {}).items():
        parser = getattr(column, "parser", None)
        if not isinstance(parser, _SystemOneParser):
            continue
        ctx = column.build_parse_context(name, schema)
        batches.setdefault(parser.batch_key(ctx), []).append((parser, ctx))
    return list(batches.values())


def questions(target: Any) -> dict[str, Question]:
    """Compile a schema's questions without contacting a provider.

    The inference rules -- options from the column's type, criteria from enum
    member docstrings, the question from its description -- are all applied
    here, so what the model will be asked is reviewable in a test with no
    credentials::

        system_one.questions(Triage)
        #> {'department': Choice(instructions='Which team should ...', ...)}
    """
    schema = _as_schema(target)
    compiled: dict[str, Question] = {}
    for group in _groups(schema):
        for parser, ctx in group:
            if isinstance(parser, Confidence):
                # reports another column's confidence; asks nothing
                continue
            compiled[ctx.target] = parser.to_question(ctx)
    return compiled


def plan(target: Any, data: Any = None, *, rows: int | None = None) -> Plan:
    """Report what validating ``data`` would send, without sending it.

    Pass a dataframe to estimate token counts from the real states, or just
    ``rows=`` for a rough count.
    """
    schema = _as_schema(target)
    groups = _groups(schema)
    asking = [
        [
            (parser, ctx)
            for parser, ctx in group
            if not isinstance(parser, Confidence)
        ]
        for group in groups
    ]
    asking = [group for group in asking if group]

    if data is not None:
        row_count = len(data)
    elif rows is not None:
        row_count = rows
    else:
        row_count = 0

    question_count = sum(len(group) for group in asking)
    request_count = len(asking) * row_count

    tokens = 0
    if data is not None:
        from pandera.system_one.execution import estimate_tokens

        for group in asking:
            _, ctx = group[0]
            for state in _build_states(data, ctx.source):
                tokens += estimate_tokens(state)

    return Plan(
        rows=row_count,
        batches=len(asking),
        requests=request_count,
        questions=question_count,
        estimated_input_tokens=tokens,
        estimated_cost_usd=None,
    )
