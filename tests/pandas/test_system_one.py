"""Tests for ``pandera.system_one``.

No test here touches the network or needs an API key: providers are either the
deterministic ``MockProvider`` or a local spy that records what it was asked.
"""

import enum
import typing

import pandas as pd
import pytest

import pandera.pandas as pa
import pandera.system_one as system_one
from pandera.api.parsers import ParseContext
from pandera.errors import SchemaError, SchemaErrors, SchemaInitError
from pandera.system_one import questions as q
from pandera.system_one.execution import TokenBucket, run_sync
from pandera.system_one.providers.base import (
    PROVIDER_ENV_VAR,
    SystemOneConfigError,
)


class Department(enum.StrEnum):
    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""
    sales = "sales"
    """Pricing, plans or account expansion."""


class Frustration(enum.IntEnum):
    calm = 0
    """Calm, simply stating facts."""
    annoyed = 1
    """Frustrated but civil."""
    angry = 2
    """Very angry, strong language."""


class Undocumented(enum.StrEnum):
    a = "a"
    b = "b"


class _Spy:
    """Records the questions it is compiled with, then answers deterministically."""

    id = "spy"
    model_version = "spy-1"

    def __init__(self, answers=None):
        self.compiled = []
        self.states = []
        self.answers = answers or {}

    @property
    def limits(self):
        return system_one.ProviderLimits(max_concurrency=4)

    def compile(self, questions):
        self.compiled.append(dict(questions))
        return dict(questions)

    async def decide(self, state, compiled):
        self.states.append(state)
        out = {}
        for name, question in compiled.items():
            if name in self.answers:
                out[name] = q.Decision(value=self.answers[name])
            elif isinstance(question, q.Choice):
                out[name] = q.Decision(
                    value=question.options[0], confidence=0.9
                )
            elif isinstance(question, q.Score):
                out[name] = q.Decision(value=0.0, confidence=0.9)
            else:
                out[name] = q.Decision(value=1.0)
        return out

    @property
    def questions(self):
        assert self.compiled, "provider was never asked to compile"
        return self.compiled[-1]


def _triage_model(**parsers):
    fields = {
        "__annotations__": {"body": str},
        "Config": type("Config", (), {"parser_source": "body"}),
    }
    for name, (annotation, parser) in parsers.items():
        fields["__annotations__"][name] = annotation
        fields[name] = parser
    return type("Model", (pa.DataFrameModel,), fields)


# --------------------------------------------------------------------------
# provider configuration
# --------------------------------------------------------------------------


def test_no_provider_raises_with_guidance():
    Model = _triage_model(
        is_urgent=(
            bool,
            pa.ParsedField(description="urgent?", parser=system_one.Noul()),
        )
    )
    with pytest.raises(SystemOneConfigError) as excinfo:
        Model.validate(pd.DataFrame({"body": ["x"]}))

    message = str(excinfo.value)
    assert "is_urgent" in message  # which column
    assert "set_provider" in message  # how to fix it


def test_provider_context_manager_scopes_and_restores():
    assert system_one.get_provider() is None
    spy = _Spy()
    with system_one.provider(spy):
        assert system_one.get_provider() is spy
    assert system_one.get_provider() is None


def test_set_provider_and_unset():
    spy = _Spy()
    system_one.set_provider(spy)
    try:
        assert system_one.get_provider() is spy
    finally:
        system_one.set_provider(None)
    assert system_one.get_provider() is None


def test_provider_from_env_var(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV_VAR, "mock:3")
    resolved = system_one.get_provider()
    assert isinstance(resolved, system_one.MockProvider)
    assert resolved.seed == 3


def test_unknown_provider_string():
    with pytest.raises(SystemOneConfigError, match="unknown System One"):
        system_one.set_provider("nope:1")


def test_provider_string_builds_typesafe_provider():
    provider = system_one.get_provider()
    assert provider is None
    with system_one.provider("typesafe:jev-1.13.0") as resolved:
        assert resolved.id == "typesafe:jev-1.13.0"
        assert resolved.model_version == "jev-1.13.0"


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def test_choice_infers_options_and_criteria_from_the_enum():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(
                description="Which team should handle this ticket",
                parser=system_one.Choice(),
            ),
        )
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["x"]}))

    question = spy.questions["department"]
    assert question.instructions == "Which team should handle this ticket"
    assert question.options == ("billing", "technical", "sales")
    assert dict(question.criteria) == {
        "billing": "Payment, invoices or subscription issues.",
        "technical": "Bugs, outages or integration problems.",
        "sales": "Pricing, plans or account expansion.",
    }


def test_score_infers_rubric_in_level_order():
    Model = _triage_model(
        frustration=(
            Frustration,
            pa.ParsedField(
                description="How frustrated the customer appears",
                parser=system_one.Score(),
            ),
        )
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["x"]}))

    question = spy.questions["frustration"]
    assert question.instructions == "How frustrated the customer appears"
    assert question.criteria == (
        "Calm, simply stating facts.",
        "Frustrated but civil.",
        "Very angry, strong language.",
    )
    assert question.levels == (0, 1, 2)


def test_explicit_instructions_and_criteria_win():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(
                description="ignored",
                parser=system_one.Choice(
                    instructions="explicit question",
                    criteria={"billing": "money things"},
                ),
            ),
        )
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["x"]}))

    question = spy.questions["department"]
    assert question.instructions == "explicit question"
    assert dict(question.criteria)["billing"] == "money things"
    # options not mentioned keep their inferred description
    assert dict(question.criteria)["sales"] == (
        "Pricing, plans or account expansion."
    )


def test_criteria_naming_an_unknown_option_raises():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(
                description="q",
                parser=system_one.Choice(criteria={"nope": "x"}),
            ),
        )
    )
    with system_one.provider(_Spy()):
        with pytest.raises(SchemaInitError, match="not among the column"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_undocumented_enum_falls_back_to_member_names():
    Model = _triage_model(
        choice=(
            Undocumented,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["x"]}))
    question = spy.questions["choice"]
    assert question.options == ("a", "b")
    # nothing to describe them with, so the model sees bare option names
    assert dict(question.criteria) == {"a": None, "b": None}


def test_missing_question_text_raises():
    Model = _triage_model(
        is_urgent=(bool, pa.ParsedField(parser=system_one.Noul()))
    )
    with system_one.provider(_Spy()):
        with pytest.raises(SchemaInitError, match="no question to ask"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_literal_column_supplies_choice_options():
    Model = _triage_model(
        routed=(
            typing.Literal["billing", "technical"],
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["x"]}))
    assert spy.questions["routed"].options == ("billing", "technical")


# --------------------------------------------------------------------------
# type compatibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parser,annotation,ok",
    [
        (system_one.Choice, Department, True),
        (system_one.Choice, Frustration, True),
        (system_one.Choice, bool, False),
        (system_one.Choice, str, False),
        (system_one.Choice, int, False),
        (system_one.Score, Frustration, True),
        (system_one.Score, Department, False),  # unordered
        (system_one.Score, bool, False),
        (system_one.Noul, bool, True),
        (system_one.Noul, float, True),
        (system_one.Noul, Department, False),
        (system_one.Noul, str, False),
    ],
)
def test_type_compatibility(parser, annotation, ok):
    Model = _triage_model(
        col=(annotation, pa.ParsedField(description="q", parser=parser()))
    )
    with system_one.provider(_Spy()):
        if ok:
            Model.validate(pd.DataFrame({"body": ["x"]}))
        else:
            with pytest.raises(SchemaInitError):
                Model.validate(pd.DataFrame({"body": ["x"]}))


def test_score_on_unordered_type_explains_the_fix():
    Model = _triage_model(
        col=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Score()),
        )
    )
    with system_one.provider(_Spy()):
        with pytest.raises(SchemaInitError) as excinfo:
            Model.validate(pd.DataFrame({"body": ["x"]}))
    message = str(excinfo.value)
    assert "unordered" in message
    assert "Choice()" in message  # the alternative


# --------------------------------------------------------------------------
# answers
# --------------------------------------------------------------------------


def test_noul_thresholds_for_a_bool_column():
    Model = _triage_model(
        flag=(
            bool,
            pa.ParsedField(
                description="q", parser=system_one.Noul(threshold=0.8)
            ),
        )
    )
    with system_one.provider(_Spy(answers={"flag": 0.9})):
        assert Model.validate(pd.DataFrame({"body": ["x"]}))[
            "flag"
        ].tolist() == [True]
    with system_one.provider(_Spy(answers={"flag": 0.7})):
        assert Model.validate(pd.DataFrame({"body": ["x"]}))[
            "flag"
        ].tolist() == [False]


def test_noul_keeps_the_probability_for_a_float_column():
    Model = _triage_model(
        score=(
            float,
            pa.ParsedField(description="q", parser=system_one.Noul()),
        )
    )
    with system_one.provider(_Spy(answers={"score": 0.42})):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert out["score"].tolist() == [0.42]


def test_score_snaps_to_the_nearest_level():
    Model = _triage_model(
        level=(
            Frustration,
            pa.ParsedField(description="q", parser=system_one.Score()),
        )
    )
    with system_one.provider(_Spy(answers={"level": 1.7})):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert out["level"].tolist() == [2]


def test_score_keeps_the_raw_position_for_a_float_column():
    class Model(pa.DataFrameModel):
        body: str
        level: float = pa.ParsedField(
            description="q",
            parser=system_one.Score(criteria=["low", "mid", "high"]),
            source="body",
        )

    # a float column has no levels of its own, so the rubric must be explicit
    with system_one.provider(_Spy(answers={"level": 1.7})):
        with pytest.raises(SchemaInitError, match="ordered categorical"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_choice_maps_labels_back_to_the_columns_values():
    """Answers arrive as strings; an int-valued option set must come back as
    ints, not as the string the wire used."""
    Model = _triage_model(
        level=(
            Frustration,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    with system_one.provider(_Spy(answers={"level": "2"})):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert out["level"].tolist() == [2]


def test_answers_land_in_the_declared_dtype():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    with system_one.provider(_Spy()):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert str(out["department"].dtype) == "category"


def test_checks_run_on_answered_columns():
    Model = _triage_model(
        level=(
            Frustration,
            pa.ParsedField(description="q", parser=system_one.Score(), ge=2),
        )
    )
    with system_one.provider(_Spy(answers={"level": 0.0})):
        with pytest.raises((SchemaError, SchemaErrors)):
            Model.validate(pd.DataFrame({"body": ["x"]}))


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_columns_sharing_a_source_are_one_request_per_row():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="a", parser=system_one.Choice()),
        ),
        frustration=(
            Frustration,
            pa.ParsedField(description="b", parser=system_one.Score()),
        ),
        is_urgent=(
            bool,
            pa.ParsedField(description="c", parser=system_one.Noul()),
        ),
    )
    provider = system_one.MockProvider()
    with system_one.provider(provider):
        Model.validate(pd.DataFrame({"body": ["x", "y", "z"]}))
    # 3 rows x 3 columns, but one request per row
    assert provider.calls == 3


def test_distinct_sources_are_distinct_batches():
    class Model(pa.DataFrameModel):
        a: str
        b: str
        from_a: bool = pa.ParsedField(
            description="q1", parser=system_one.Noul(), source="a"
        )
        from_b: bool = pa.ParsedField(
            description="q2", parser=system_one.Noul(), source="b"
        )

    provider = system_one.MockProvider()
    with system_one.provider(provider):
        Model.validate(pd.DataFrame({"a": ["x"], "b": ["y"]}))
    assert provider.calls == 2


def test_one_batch_asks_every_question_together():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="a", parser=system_one.Choice()),
        ),
        is_urgent=(
            bool,
            pa.ParsedField(description="c", parser=system_one.Noul()),
        ),
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["x"]}))
    assert set(spy.questions) == {"department", "is_urgent"}


def test_multi_column_source_becomes_a_mapping_state():
    class Model(pa.DataFrameModel):
        subject: str
        body: str
        urgent: bool = pa.ParsedField(
            description="q",
            parser=system_one.Noul(),
            source=["subject", "body"],
        )

    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"subject": ["hi"], "body": ["there"]}))
    assert spy.states == [{"subject": "hi", "body": "there"}]


def test_single_column_source_sends_the_value_itself():
    Model = _triage_model(
        urgent=(
            bool,
            pa.ParsedField(description="q", parser=system_one.Noul()),
        )
    )
    spy = _Spy()
    with system_one.provider(spy):
        Model.validate(pd.DataFrame({"body": ["hello"]}))
    assert spy.states == ["hello"]


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


def test_mock_provider_is_deterministic():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    frame = pd.DataFrame({"body": ["a", "b", "c"]})
    with system_one.provider(system_one.MockProvider(seed=1)):
        first = Model.validate(frame)["department"].tolist()
    with system_one.provider(system_one.MockProvider(seed=1)):
        second = Model.validate(frame)["department"].tolist()
    assert first == second


def test_mock_provider_seed_changes_answers():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    frame = pd.DataFrame({"body": [str(i) for i in range(20)]})
    with system_one.provider(system_one.MockProvider(seed=1)):
        first = Model.validate(frame)["department"].tolist()
    with system_one.provider(system_one.MockProvider(seed=2)):
        second = Model.validate(frame)["department"].tolist()
    assert first != second


def test_record_then_replay():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    frame = pd.DataFrame({"body": ["a", "b"]})

    recorder = system_one.RecordingProvider(system_one.MockProvider(seed=5))
    with system_one.provider(recorder):
        recorded = Model.validate(frame)["department"].tolist()

    replay = system_one.ReplayProvider(
        recorder.cassette, id=recorder.id, model_version=recorder.model_version
    )
    with system_one.provider(replay):
        replayed = Model.validate(frame)["department"].tolist()

    assert recorded == replayed


def test_replay_without_a_recording_fails_loudly():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    with system_one.provider(system_one.ReplayProvider({})):
        with pytest.raises(KeyError, match="no recorded answer"):
            Model.validate(pd.DataFrame({"body": ["unseen"]}))


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def test_answers_stay_aligned_with_rows():
    """Requests complete out of order; answers must not."""

    class _Shuffled:
        id = "shuffled"
        model_version = "1"

        @property
        def limits(self):
            return system_one.ProviderLimits(max_concurrency=8)

        def compile(self, questions):
            return dict(questions)

        async def decide(self, state, compiled):
            import asyncio

            # later rows finish first
            await asyncio.sleep((10 - int(state)) / 1000)
            return {"n": q.Decision(value=float(state) / 10)}

    class Model(pa.DataFrameModel):
        body: str
        n: float = pa.ParsedField(
            description="q", parser=system_one.Noul(), source="body"
        )

    frame = pd.DataFrame({"body": [str(i) for i in range(10)]})
    with system_one.provider(_Shuffled()):
        out = Model.validate(frame)
    assert out["n"].tolist() == [i / 10 for i in range(10)]


def test_run_sync_works_inside_a_running_loop():
    import asyncio

    async def _outer():
        async def _inner():
            return 42

        return run_sync(_inner())

    assert asyncio.run(_outer()) == 42


def test_token_bucket_throttles():
    import asyncio

    bucket = TokenBucket(requests_per_minute=60)

    async def _burst():
        for _ in range(3):
            await bucket.acquire()

    asyncio.run(_burst())
    # a 60/min budget starts full at 60, so three requests do not wait
    assert bucket.waits == 0


def test_token_bucket_waits_when_exhausted():
    import asyncio

    bucket = TokenBucket(tokens_per_second=10)
    bucket._token_allowance = 0.0

    async def _one():
        await bucket.acquire(5)

    asyncio.run(_one())
    assert bucket.waits > 0


# --------------------------------------------------------------------------
# typesafe provider translation (no network)
# --------------------------------------------------------------------------


def test_typesafe_provider_translates_questions():
    pytest.importorskip("typesafe_sdk")
    import typesafe_sdk

    from pandera.system_one.providers.typesafe import TypeSafeProvider

    provider = TypeSafeProvider("jev-1.13.0")
    compiled = provider.compile(
        {
            "choice": q.Choice(
                instructions="pick",
                criteria=(("a", "first"), ("b", None)),
                values=("a", "b"),
            ),
            "score": q.Score(
                instructions="rate", criteria=("low", "high"), levels=(0, 1)
            ),
            "noul": q.Noul(instructions="yes?"),
        }
    )
    assert isinstance(compiled["choice"], typesafe_sdk.Choice)
    assert compiled["choice"].criteria == {"a": "first", "b": None}
    assert isinstance(compiled["score"], typesafe_sdk.Score)
    assert list(compiled["score"].criteria) == ["low", "high"]
    assert isinstance(compiled["noul"], typesafe_sdk.Noul)
    assert compiled["noul"].instructions == "yes?"


def test_typesafe_provider_normalizes_answers():
    pytest.importorskip("typesafe_sdk")
    from pandera.system_one.providers.typesafe import _to_decision

    class _ChoiceAnswer:
        type = "choice"
        choice = "billing"
        confidence = 0.91
        probabilities = {"billing": 0.91, "sales": 0.09}

    decision = _to_decision(_ChoiceAnswer())
    assert decision.value == "billing"
    assert decision.confidence == 0.91
    assert decision.probabilities["sales"] == 0.09

    class _NoulAnswer:
        type = "noul"
        noul = 0.77

    # a noul is a probability and carries no separate confidence
    assert _to_decision(_NoulAnswer()).value == 0.77
    assert _to_decision(_NoulAnswer()).confidence is None
