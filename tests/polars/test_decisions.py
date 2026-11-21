"""Tests for the decisions API with the polars backend.

No test here touches the network or needs an API key: providers are either
the deterministic ``MockProvider`` or a local spy that records what it was
asked. These mirror the pandas decisions tests, adapted to polars frames.
"""

import enum

import polars as pl
import pytest

import pandera.config as _config
import pandera.decisions as decisions
import pandera.polars as pa
from pandera.decisions import primitives as q
from pandera.decisions.providers.base import DecisionsConfigError
from pandera.errors import SchemaInitError

# The decisions API is implemented on the native polars backend, not the
# narwhals backend (which has no parser step for non-pandas frames).
pytestmark = pytest.mark.skipif(
    _config.CONFIG.use_narwhals_backend,
    reason="the decisions API is not supported by the narwhals backend",
)


class Department(str, enum.Enum):
    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""
    sales = "sales"
    """Pricing, plans or account expansion."""


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
        return decisions.ProviderLimits(max_concurrency=4)

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
                out[name] = q.Decision(value=question.options[0])
            elif isinstance(question, q.Score):
                out[name] = q.Decision(value=0.0)
            else:
                out[name] = q.Decision(value=1.0)
        return out

    @property
    def questions(self):
        assert self.compiled, "provider was never asked to compile"
        return self.compiled[-1]


def _triage_model(**parsers):
    fields = {
        "__annotations__": {"body": str, "ticket_id": int},
        "Config": type("Config", (), {"parser_source": "body"}),
    }
    for name, (annotation, parser) in parsers.items():
        fields["__annotations__"][name] = annotation
        fields[name] = parser
    return type("Model", (pa.DataFrameModel,), fields)


def _frame(*args, **kwargs):
    return pl.DataFrame(*args, **kwargs)


def _frame2(body):
    return _frame({"body": body, "ticket_id": [1] * len(body)})


# --------------------------------------------------------------------------
# provider configuration
# --------------------------------------------------------------------------


def test_no_provider_raises_with_guidance():
    Model = _triage_model(
        is_urgent=(
            bool,
            pa.ParsedField(description="urgent?", parser=decisions.Noul()),
        )
    )
    with pytest.raises(DecisionsConfigError) as excinfo:
        Model.validate(_frame2(["x"]))

    message = str(excinfo.value)
    assert "is_urgent" in message  # which column
    assert "set_provider" in message  # how to fix it


def test_provider_context_manager_scopes_and_restores():
    assert decisions.get_provider() is None
    spy = _Spy()
    with decisions.provider(spy):
        assert decisions.get_provider() is spy
    assert decisions.get_provider() is None


# --------------------------------------------------------------------------
# Noul
# --------------------------------------------------------------------------


def test_noul_thresholds_for_a_bool_column():
    Model = _triage_model(
        flag=(
            bool,
            pa.ParsedField(
                description="q", parser=decisions.Noul(threshold=0.8)
            ),
        )
    )
    with decisions.provider(_Spy(answers={"flag": 0.9})):
        assert Model.validate(_frame2(["x"]))["flag"].to_list() == [True]
    with decisions.provider(_Spy(answers={"flag": 0.7})):
        assert Model.validate(_frame2(["x"]))["flag"].to_list() == [False]


def test_noul_keeps_the_probability_for_a_float_column():
    Model = _triage_model(
        score=(
            float,
            pa.ParsedField(description="q", parser=decisions.Noul()),
        )
    )
    with decisions.provider(_Spy(answers={"score": 0.42})):
        out = Model.validate(_frame2(["x"]))
    assert out["score"].to_list() == [0.42]


def test_noul_rejects_a_string_column_at_validation():
    Model = _triage_model(
        bad=(
            str,
            pa.ParsedField(description="q", parser=decisions.Noul()),
        )
    )
    with decisions.provider(_Spy()):
        with pytest.raises(SchemaInitError, match="type must be"):
            Model.validate(_frame2(["x"]))


def test_mock_provider_fills_columns_without_network():
    class Model(pa.DataFrameModel):
        body: str
        is_urgent: bool = pa.ParsedField(
            description="q", parser=decisions.Noul(threshold=0.5)
        )

        class Config:
            parser_source = "body"

    with decisions.provider(decisions.MockProvider(seed=1)):
        out = Model.validate(_frame2(["hello", "world"]))
    assert out["is_urgent"].dtype == pl.Boolean
    assert set(out["is_urgent"].to_list()).issubset({True, False})


# --------------------------------------------------------------------------
# Choice
# --------------------------------------------------------------------------


def test_choice_infers_options_from_the_enum():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(
                description="hand off to", parser=decisions.Choice()
            ),
        )
    )
    with decisions.provider(_Spy()) as resolved:
        Model.validate(_frame2(["x"]))
        question = resolved.questions["department"]
    assert question.options == ("billing", "technical", "sales")


def test_choice_maps_labels_back_to_the_columns_values():
    class Model(pa.DataFrameModel):
        body: str
        department: Department = pa.ParsedField(
            description="hand off to",
            parser=decisions.Choice(),
            source="body",
        )

    with decisions.provider(_Spy(answers={"department": "billing"})):
        out = Model.validate(_frame({"body": ["x"]}))
    assert out["department"].to_list() == ["billing"]
    assert out["department"].dtype == pl.Enum


def test_choice_works_for_eager_and_lazy_frames():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(
                description="hand off to", parser=decisions.Choice()
            ),
        )
    )
    with decisions.provider(_Spy(answers={"department": "sales"})):
        eager = Model.validate(_frame2(["x"]))
        lazy = Model.validate(_frame2(["x"]).lazy()).collect()
    assert eager["department"].to_list() == ["sales"]
    assert lazy["department"].to_list() == ["sales"]


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_batched_columns_share_one_request():
    Model = _triage_model(
        department=(
            Department,
            pa.ParsedField(description="d", parser=decisions.Choice()),
        ),
        is_urgent=(
            bool,
            pa.ParsedField(description="u", parser=decisions.Noul()),
        ),
    )
    with decisions.provider(_Spy()) as resolved:
        Model.validate(_frame2(["x"]))

    # both columns share a provider + source, so one compiled request
    assert len(resolved.compiled) == 1
    assert set(resolved.questions) == {"department", "is_urgent"}
