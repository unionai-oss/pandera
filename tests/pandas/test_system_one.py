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
from pandera.system_one import primitives as q
from pandera.system_one.execution import TokenBucket, run_sync
from pandera.system_one.providers import base as provider_base
from pandera.system_one.providers.base import (
    PROVIDER_ENV_VAR,
    ProviderCapabilityError,
    SystemOneConfigError,
)


class Department(str, enum.Enum):
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


class Undocumented(str, enum.Enum):
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


# --------------------------------------------------------------------------
# providers are interchangeable: registry, capabilities, a second backend
# --------------------------------------------------------------------------


def test_register_provider_makes_a_prefix_resolvable(monkeypatch):
    monkeypatch.setattr(provider_base, "_FACTORIES", {})
    system_one.register_provider(
        "acme", lambda model: system_one.MockProvider(seed=len(model))
    )
    with system_one.provider("acme:decider-2b") as resolved:
        assert isinstance(resolved, system_one.MockProvider)
        assert resolved.seed == len("decider-2b")


def test_unknown_prefix_lists_what_is_known(monkeypatch):
    monkeypatch.setattr(provider_base, "_FACTORIES", {})
    system_one.register_provider("acme", lambda model: None)
    with pytest.raises(SystemOneConfigError) as excinfo:
        system_one.set_provider("nope:1")
    message = str(excinfo.value)
    for prefix in ("typesafe", "ollaya", "mock", "acme"):
        assert f"'{prefix}:'" in message
    assert "register_provider" in message


def test_provider_without_capabilities_is_asked_the_shared_vocabulary():
    # ``_Spy`` declares no ``capabilities`` at all
    caps = system_one.capabilities_of(_Spy())
    assert caps == system_one.ProviderCapabilities()
    assert caps.max_options == 255 and caps.max_levels == 10


def test_ollaya_provider_string_uses_local_defaults(monkeypatch):
    monkeypatch.delenv("OLLAYA_HOST", raising=False)
    with system_one.provider("ollaya:decider:2b") as resolved:
        assert resolved.id == "ollaya:decider:2b"
        assert resolved.model_version == "decider:2b"
        # a local model has a queue, not a quota
        assert resolved.limits.requests_per_minute is None
        assert resolved.limits.tokens_per_second is None
        assert resolved.limits.max_state_tokens == 65_536
        assert resolved._base_url == "http://127.0.0.1:11435"


def test_ollaya_host_env_var_sets_the_server(monkeypatch):
    monkeypatch.setenv("OLLAYA_HOST", "gpu-box:9999")
    assert system_one.OllayaProvider("laya")._base_url == "http://gpu-box:9999"
    monkeypatch.setenv("OLLAYA_HOST", "https://ollaya.internal")
    assert (
        system_one.OllayaProvider("laya")._base_url
        == "https://ollaya.internal"
    )
    assert (
        system_one.OllayaProvider("laya", base_url="http://x:1")._base_url
        == "http://x:1"
    )


@pytest.mark.parametrize(
    "model, max_options",
    [
        ("laya", 125),  # routes between en and multilingual: the smaller
        ("laya:en", 125),
        ("laya:multilingual", 250),
        ("decider:2b", 255),
    ],
)
def test_ollaya_capabilities_follow_the_model(model, max_options):
    caps = system_one.OllayaProvider(model).capabilities
    assert caps.max_options == max_options
    assert caps.max_questions == 256


class _Big(str, enum.Enum):
    a = "a"
    b = "b"
    c = "c"
    d = "d"


def _narrow(**caps):
    return system_one.MockProvider(
        capabilities=system_one.ProviderCapabilities(**caps)
    )


def test_options_over_the_providers_budget_fail_before_any_request():
    Model = _triage_model(
        big=(
            _Big,
            pa.ParsedField(description="pick", parser=system_one.Choice()),
        )
    )
    narrow = _narrow(max_options=3)
    with system_one.provider(narrow):
        with pytest.raises(ProviderCapabilityError) as excinfo:
            Model.validate(pd.DataFrame({"body": ["x"]}))

    message = str(excinfo.value)
    assert "'big'" in message  # which column
    assert "4 options" in message and "at most 3" in message  # what limit
    assert "mock:0" in message  # which provider
    assert narrow.calls == 0  # nothing was sent


def test_a_question_kind_the_model_does_not_answer_is_refused():
    Model = _triage_model(
        is_urgent=(
            bool,
            pa.ParsedField(description="urgent?", parser=system_one.Noul()),
        )
    )
    classifier_only = _narrow(kinds=frozenset({"choice"}))
    with system_one.provider(classifier_only):
        with pytest.raises(ProviderCapabilityError, match="noul question"):
            Model.validate(pd.DataFrame({"body": ["x"]}))
    assert classifier_only.calls == 0


def test_too_many_questions_for_one_request_is_refused():
    Model = _triage_model(
        a=(bool, pa.ParsedField(description="a?", parser=system_one.Noul())),
        b=(bool, pa.ParsedField(description="b?", parser=system_one.Noul())),
    )
    with system_one.provider(_narrow(max_questions=1)):
        with pytest.raises(ProviderCapabilityError, match="at most 1"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_score_levels_over_the_providers_budget_are_refused():
    Model = _triage_model(
        frustration=(
            Frustration,
            pa.ParsedField(description="mood", parser=system_one.Score()),
        )
    )
    with system_one.provider(_narrow(max_levels=2)):
        with pytest.raises(ProviderCapabilityError, match="3 levels"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_state_over_the_providers_limit_names_the_row():
    Model = _triage_model(
        is_urgent=(
            bool,
            pa.ParsedField(description="urgent?", parser=system_one.Noul()),
        )
    )
    small = system_one.MockProvider(
        limits=system_one.ProviderLimits(max_state_tokens=10)
    )
    frame = pd.DataFrame({"body": ["short", "x" * 400]})
    with system_one.provider(small):
        with pytest.raises(Exception, match="row 1") as excinfo:
            Model.validate(frame)
    assert "tokens" in str(excinfo.value)


# A response as Ollaya serves it. The fields the TypeSafe SDK does not model --
# ``routing``, ``state_truncated``, timings -- are there on purpose: a real
# server sends them.
def _ollaya_response(request):
    import json

    import httpx2

    body = json.loads(request.content)
    assert str(request.url) == "http://127.0.0.1:11435/v1/systemone"
    assert request.headers["authorization"] == "Bearer local"
    assert body["model"] == "laya"
    answers = {}
    for name, question in body["questions"].items():
        if question["type"] == "choice":
            options = list(question["criteria"])
            answers[name] = {
                "type": "choice",
                "choice": options[1],
                "confidence": 0.9547,
                "probabilities": {
                    option: 0.5 if i == 1 else 0.5 / (len(options) - 1)
                    for i, option in enumerate(options)
                },
            }
        elif question["type"] == "score":
            answers[name] = {
                "type": "score",
                "score": 1.5234,
                "confidence": 0.4521,
                "legend": {
                    str(i): text for i, text in enumerate(question["criteria"])
                },
                "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
            }
        else:
            # a noul is a bare probability: no confidence, no distribution
            answers[name] = {"type": "noul", "noul": 0.9127}
    return httpx2.Response(
        200,
        json={
            "model": "laya:en",
            "answers": answers,
            "usage": {"input_tokens": 118, "output_tokens": 0},
            "routing": {"router": "laya:latest", "model": "laya:en"},
            "state_truncated": False,
            "done_reason": "decide",
        },
    )


def test_ollaya_fills_a_schema_end_to_end_through_the_sdk():
    """The same schema, a different provider: no edits, no vendor code."""
    pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")

    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="team", parser=system_one.Choice()),
        ),
        frustration=(
            Frustration,
            pa.ParsedField(description="mood", parser=system_one.Score()),
        ),
        is_urgent=(
            bool,
            pa.ParsedField(description="urgent?", parser=system_one.Noul()),
        ),
    )
    ollaya = system_one.OllayaProvider(
        "laya", transport=httpx2.MockTransport(_ollaya_response)
    )
    with system_one.provider(ollaya):
        out = Model.validate(pd.DataFrame({"body": ["a", "b"]}))

    assert list(out["department"]) == ["technical", "technical"]
    assert list(out["frustration"]) == [Frustration.angry] * 2  # 1.52 -> 2
    assert list(out["is_urgent"]) == [True, True]
