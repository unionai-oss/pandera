"""Semantic checks.

Judging a column that already exists is a :class:`~pandera.api.checks.Check`,
not a parser -- it produces a verdict about values rather than the values
themselves. Being an ordinary ``Check`` means failure cases, ``lazy=True``,
``n_failure_cases`` and ``raise_warning`` all work untouched.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd

from pandera.api.checks import Check
from pandera.errors import SchemaInitError
from pandera.system_one import primitives as q
from pandera.system_one.execution import gather_decisions, run_sync
from pandera.system_one.providers import base as provider_base

_ANSWER = "holds"


def Holds(
    instructions: str,
    *,
    context: Sequence[str] | None = None,
    min_probability: float = 0.5,
    provider: Any = None,
    name: str | None = None,
    **check_kwargs: Any,
) -> Check:
    """Check that a statement holds of each row, as judged by a model.

    ::

        schema = pa.DataFrameSchema(
            {
                "name": pa.Column(str),
                "category": pa.Column(str),
                "description": pa.Column(str),
            },
            checks=system_one.Holds(
                "The description is a coherent description of a product "
                "belonging to the stated category",
                context=["name", "category", "description"],
                min_probability=0.85,
            ),
        )

    Used as a column check, the column's own value is what gets judged. Used as
    a dataframe check with ``context``, those columns are sent instead, which
    is the only way to judge a value *relative to* another column.

    Inside a ``@pa.dataframe_check`` method, use :func:`holds` instead -- that
    wants the boolean Series, not a ``Check``.

    :param instructions: the statement to judge.
    :param context: columns to send. Requires a dataframe-level check.
    :param min_probability: the probability at or above which the statement is
        taken to hold.
    :param provider: overrides the configured provider.
    :param check_kwargs: everything :class:`~pandera.api.checks.Check` accepts.
    """
    _judge = holds(
        instructions,
        context=context,
        min_probability=min_probability,
        provider=provider,
        name=name,
    )
    return Check(
        _judge,
        name=name or "holds",
        description=instructions,
        **check_kwargs,
    )


def holds(
    instructions: str,
    *,
    context: Sequence[str] | None = None,
    min_probability: float = 0.5,
    provider: Any = None,
    name: str | None = None,
) -> Any:
    """The predicate behind :func:`Holds`, for use inside ``@dataframe_check``.

    ``Holds`` returns a ``Check``, which is what ``DataFrameSchema(checks=...)``
    and ``Field(checks=...)`` want. A ``@pa.dataframe_check`` method wants the
    boolean Series itself::

        @pa.dataframe_check
        def name_fits_category(cls, df):
            return system_one.holds(
                "The name fits the category",
                context=["name", "category"],
            )(df)
    """
    question = {_ANSWER: q.Noul(instructions=instructions)}
    sources = tuple(context) if context else None

    def _judge(obj: Any) -> Any:
        if not provider_base.enabled():
            # Schemas carrying semantic checks still have to run offline --
            # in CI, on a plane, in a test suite with no credentials.
            warnings.warn(
                f"System One checks are disabled, so {instructions!r} was "
                f"not evaluated. Unset {provider_base.ENABLED_ENV_VAR} to "
                "enable them.",
                stacklevel=2,
            )
            return _all_true(obj)

        resolved = (
            provider_base._coerce(provider)
            if provider is not None
            else provider_base.require_provider([name or "semantic check"])
        )
        states = _states(obj, sources, instructions)
        provider_base.verify_questions(question, resolved)
        prepared = resolved.compile(question)

        async def _decide(state: Any) -> Any:
            return await resolved.decide(state, prepared)

        answers = run_sync(gather_decisions(states, _decide, resolved.limits))
        verdicts = [
            False
            if answer is None
            else float(answer[_ANSWER].value) >= min_probability
            for answer in answers
        ]
        return pd.Series(verdicts, index=_index(obj))

    return _judge


def _index(obj: Any) -> Any:
    return obj.index


def _all_true(obj: Any) -> Any:
    return pd.Series(True, index=_index(obj))


def _states(
    obj: Any, sources: tuple[str, ...] | None, instructions: str
) -> list[Any]:
    """One state per row: the column's values, or the context columns."""
    if sources is None:
        if isinstance(obj, pd.DataFrame):
            raise SchemaInitError(
                f"the semantic check {instructions!r} is applied to a "
                "dataframe but names no `context` columns, so there is "
                "nothing specific to judge. Pass context=[...], or attach "
                "the check to a single column."
            )
        return [_jsonable(value) for value in obj.tolist()]

    if not isinstance(obj, pd.DataFrame):
        raise SchemaInitError(
            f"the semantic check {instructions!r} names `context` columns "
            f"{list(sources)}, which requires a dataframe-level check. "
            "Attach it with @pa.dataframe_check or DataFrameSchema(checks=...)."
        )

    missing = [column for column in sources if column not in obj.columns]
    if missing:
        raise SchemaInitError(
            f"the semantic check {instructions!r} names `context` columns "
            f"{missing}, which are not in the dataframe."
        )
    return [
        {column: _jsonable(row[column]) for column in sources}
        for _, row in obj[list(sources)].iterrows()
    ]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
