"""Confidence columns, abstention, caching, stats and inspection."""

import enum

import pandas as pd
import pytest

import pandera.pandas as pa
import pandera.system_one as system_one
from pandera.errors import SchemaError, SchemaErrors, SchemaInitError
from pandera.system_one import primitives as q
from pandera.system_one.cache import MemoryCache, SQLiteCache, build_cache
from pandera.system_one.providers.base import ProviderCapabilityError


class Department(str, enum.Enum):
    billing = "billing"
    """Payment, invoices or subscription issues."""
    technical = "technical"
    """Bugs, outages or integration problems."""


class _Fixed:
    """Answers with a fixed value and confidence, counting its calls."""

    id = "fixed"
    model_version = "fixed-1"

    def __init__(self, value="billing", confidence=0.9):
        self.value = value
        self.confidence = confidence
        self.calls = 0

    @property
    def limits(self):
        return system_one.ProviderLimits(max_concurrency=4)

    def compile(self, questions):
        return dict(questions)

    async def decide(self, state, compiled):
        self.calls += 1
        return {
            name: q.Decision(value=self.value, confidence=self.confidence)
            for name in compiled
        }


def _model(**extra_fields):
    fields = {
        "__annotations__": {"body": str},
        "Config": type("Config", (), {"parser_source": "body"}),
    }
    for name, (annotation, parser) in extra_fields.items():
        fields["__annotations__"][name] = annotation
        fields[name] = parser
    return type("Model", (pa.DataFrameModel,), fields)


# --------------------------------------------------------------------------
# confidence columns
# --------------------------------------------------------------------------


def test_confidence_column_reports_a_siblings_confidence():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        ),
        department_confidence=(
            float,
            pa.ParsedField(parser=system_one.Confidence("department")),
        ),
    )
    with system_one.provider(_Fixed(confidence=0.83)):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert out["department_confidence"].tolist() == [0.83]


def test_confidence_column_is_free():
    """It reads a decision a sibling already paid for, so it adds no request."""
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        ),
        department_confidence=(
            float,
            pa.ParsedField(parser=system_one.Confidence("department")),
        ),
    )
    provider = _Fixed()
    with system_one.provider(provider):
        Model.validate(pd.DataFrame({"body": ["a", "b", "c"]}))
    assert provider.calls == 3  # one per row, not two per row


def test_confidence_column_is_checked_like_any_other():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        ),
        department_confidence=(
            float,
            pa.ParsedField(parser=system_one.Confidence("department"), ge=0.9),
        ),
    )
    with system_one.provider(_Fixed(confidence=0.5)):
        with pytest.raises((SchemaError, SchemaErrors)):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_confidence_of_a_column_in_another_batch_is_an_error():
    class Model(pa.DataFrameModel):
        a: str
        b: str
        department: Department = pa.ParsedField(
            description="q", parser=system_one.Choice(), source="a"
        )
        department_confidence: float = pa.ParsedField(
            parser=system_one.Confidence("department"), source="b"
        )

    with system_one.provider(_Fixed()):
        with pytest.raises(SchemaInitError, match="not answered in the same"):
            Model.validate(pd.DataFrame({"a": ["x"], "b": ["y"]}))


def test_confidence_of_an_unknown_column_is_an_error():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        ),
        conf=(float, pa.ParsedField(parser=system_one.Confidence("nope"))),
    )
    with system_one.provider(_Fixed()):
        with pytest.raises(SchemaInitError, match="'nope'"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


# --------------------------------------------------------------------------
# abstention
# --------------------------------------------------------------------------


def test_abstain_below_nulls_low_confidence_answers():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="q",
                parser=system_one.Choice(abstain_below=0.8),
                nullable=True,
            ),
        )
    )
    with system_one.provider(_Fixed(confidence=0.9)):
        assert Model.validate(pd.DataFrame({"body": ["x"]}))[
            "department"
        ].tolist() == ["billing"]

    with system_one.provider(_Fixed(confidence=0.5)):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert out["department"].isna().all()


def test_abstention_composes_with_a_confidence_floor():
    """Two independent levers: rows below the floor become null, and the
    abstention *rate* is then a check on the frame."""

    class Model(pa.DataFrameModel):
        body: str
        department: Department = pa.ParsedField(
            description="q",
            parser=system_one.Choice(abstain_below=0.8),
            nullable=True,
            source="body",
        )

        @pa.dataframe_check
        def not_too_many_abstentions(cls, df):
            return df["department"].isna().mean() < 0.5

    with system_one.provider(_Fixed(confidence=0.5)):
        with pytest.raises((SchemaError, SchemaErrors)):
            Model.validate(pd.DataFrame({"body": ["x", "y"]}))


def test_abstaining_on_a_noul_is_refused_not_silently_ignored():
    """A noul is a bare probability: no provider reports a confidence for it.

    ``abstain_below`` on one used to be a no-op -- the safety floor the user
    asked for simply never fired. It is now an error naming the fix.
    """
    Model = _model(
        flag=(
            bool,
            pa.ParsedField(
                description="q",
                parser=system_one.Noul(abstain_below=0.8),
                nullable=True,
            ),
        )
    )
    provider = _Fixed(value=0.9, confidence=None)
    with system_one.provider(provider):
        with pytest.raises(ProviderCapabilityError) as excinfo:
            Model.validate(pd.DataFrame({"body": ["x"]}))
    assert "'flag'" in str(excinfo.value)
    assert "calibrated probability" in str(excinfo.value)  # the way out
    assert provider.calls == 0  # refused before any request


def test_confidence_of_a_noul_is_refused():
    Model = _model(
        flag=(
            bool,
            pa.ParsedField(description="q", parser=system_one.Noul()),
        ),
        flag_confidence=(
            float,
            pa.ParsedField(parser=system_one.Confidence("flag")),
        ),
    )
    with system_one.provider(_Fixed(confidence=None)):
        with pytest.raises(ProviderCapabilityError, match="'flag'"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


class _ReportsNoulConfidence(_Fixed):
    """A model that *does* report a confidence for noul questions."""

    @property
    def capabilities(self):
        return system_one.ProviderCapabilities(
            reports_confidence=frozenset({"choice", "score", "noul"})
        )


def test_a_provider_that_reports_noul_confidence_can_abstain_on_it():
    Model = _model(
        flag=(
            pd.BooleanDtype,
            pa.ParsedField(
                description="q",
                parser=system_one.Noul(abstain_below=0.8),
                nullable=True,
            ),
        )
    )
    with system_one.provider(
        _ReportsNoulConfidence(value=0.9, confidence=0.5)
    ):
        out = Model.validate(pd.DataFrame({"body": ["x"]}))
    assert out["flag"].isna().all()


def test_an_abstained_noul_is_missing_not_false():
    """``astype(bool)`` maps a missing answer to False -- a confident "no"."""
    Model = _model(
        flag=(
            bool,
            pa.ParsedField(
                description="q",
                parser=system_one.Noul(abstain_below=0.8),
                nullable=True,
            ),
        )
    )
    with system_one.provider(
        _ReportsNoulConfidence(value=0.9, confidence=0.5)
    ):
        # a plain ``bool`` column cannot hold a missing answer: it says so
        with pytest.raises((SchemaError, SchemaErrors), match="boolean|bool"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


def test_a_provider_that_promises_confidence_and_omits_it_is_an_error():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="q",
                parser=system_one.Choice(abstain_below=0.8),
                nullable=True,
            ),
        )
    )
    # declares confidence for choices (the default), then does not send one
    with system_one.provider(_Fixed(confidence=None)):
        with pytest.raises(ProviderCapabilityError, match="no confidence"):
            Model.validate(pd.DataFrame({"body": ["x"]}))


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------


def test_cache_avoids_repeat_requests():
    cache = MemoryCache()
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="q",
                parser=system_one.Choice(cache=cache),
            ),
        )
    )
    frame = pd.DataFrame({"body": ["a", "b"]})

    provider = _Fixed()
    with system_one.provider(provider):
        Model.validate(frame)
        assert provider.calls == 2
        Model.validate(frame)
        assert provider.calls == 2  # served from cache


def test_cache_key_includes_the_question_text():
    """Rewording a description must invalidate its cached answers -- the
    payoff for keeping the question in the schema rather than in a prompt."""
    from pandera.system_one.cache import cache_key

    def _key(instructions):
        return cache_key(
            "p",
            "v1",
            "some state",
            {
                "col": q.Choice(
                    instructions=instructions,
                    criteria=(("a", None), ("b", None)),
                    values=("a", "b"),
                )
            },
        )

    assert _key("first wording") == _key("first wording")
    assert _key("first wording") != _key("different wording")


def test_cache_key_includes_criteria_and_model_version():
    from pandera.system_one.cache import cache_key

    def _key(*, criteria, model="v1"):
        return cache_key(
            "p",
            model,
            "state",
            {
                "col": q.Choice(
                    instructions="q", criteria=criteria, values=("a", "b")
                )
            },
        )

    base = (("a", "first"), ("b", "second"))
    assert _key(criteria=base) == _key(criteria=base)
    # a reworded option description is a different question
    assert _key(criteria=base) != _key(
        criteria=(("a", "reworded"), ("b", "second"))
    )
    # so is the same question put to a different model version
    assert _key(criteria=base) != _key(criteria=base, model="v2")


def test_an_in_memory_cache_is_not_shared_between_schemas():
    """Pandera deep-copies columns into a schema, so a cache instance handed
    to one model is not the object another model ends up using. A shared
    cache needs shared storage -- SQLite, or a custom cache over a global."""
    cache = MemoryCache()

    def _build():
        return _model(
            department=(
                Department,
                pa.ParsedField(
                    description="q", parser=system_one.Choice(cache=cache)
                ),
            )
        )

    frame = pd.DataFrame({"body": ["a"]})
    provider = _Fixed()
    with system_one.provider(provider):
        _build().validate(frame)
        _build().validate(frame)
    assert provider.calls == 2  # each schema kept its own copy
    assert len(cache) == 0  # and neither wrote to the original


def test_cache_partial_hit_only_calls_for_new_rows():
    cache = MemoryCache()
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="q", parser=system_one.Choice(cache=cache)
            ),
        )
    )
    provider = _Fixed()
    with system_one.provider(provider):
        Model.validate(pd.DataFrame({"body": ["a"]}))
        assert provider.calls == 1
        # yesterday's row plus a new one
        Model.validate(pd.DataFrame({"body": ["a", "b"]}))
        assert provider.calls == 2


def test_sqlite_cache_survives_a_new_instance(tmp_path):
    path = tmp_path / "answers.db"
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="q",
                parser=system_one.Choice(cache=f"sqlite:///{path}"),
            ),
        )
    )
    frame = pd.DataFrame({"body": ["a"]})
    provider = _Fixed()
    with system_one.provider(provider):
        Model.validate(frame)
        assert provider.calls == 1
        Model.validate(frame)
        assert provider.calls == 1


def test_build_cache_forms():
    assert isinstance(build_cache("memory"), MemoryCache)
    assert build_cache(None) is None
    custom = MemoryCache()
    assert build_cache(custom) is custom
    with pytest.raises(ValueError, match="unknown cache"):
        build_cache("redis://nope")


def test_sqlite_cache_roundtrips_a_decision(tmp_path):
    cache = SQLiteCache(str(tmp_path / "c.db"))
    assert cache.get("missing") is None
    cache.set(
        "k",
        {
            "col": q.Decision(
                value="billing", confidence=0.9, probabilities={"billing": 0.9}
            )
        },
    )
    restored = cache.get("k")
    assert restored["col"].value == "billing"
    assert restored["col"].confidence == 0.9
    assert restored["col"].probabilities["billing"] == 0.9


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def test_stats_are_attached_to_the_validated_frame():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        )
    )
    with system_one.provider(_Fixed()):
        out = Model.validate(pd.DataFrame({"body": ["a", "b", "c"]}))

    stats = system_one.stats(out)
    assert stats["rows"] == 3
    assert stats["batches"] == 1
    assert stats["called"] == 3
    assert stats["cached"] == 0
    assert stats["provider"] == "fixed"
    assert stats["model_version"] == "fixed-1"
    assert stats["seconds"] >= 0


def test_stats_report_cache_hits():
    cache = MemoryCache()
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="q", parser=system_one.Choice(cache=cache)
            ),
        )
    )
    frame = pd.DataFrame({"body": ["a", "b"]})
    with system_one.provider(_Fixed()):
        Model.validate(frame)
        out = Model.validate(frame)
    stats = system_one.stats(out)
    assert stats["cached"] == 2
    assert stats["called"] == 0


def test_stats_accumulate_across_batches():
    class Model(pa.DataFrameModel):
        a: str
        b: str
        from_a: bool = pa.ParsedField(
            description="q1", parser=system_one.Noul(), source="a"
        )
        from_b: bool = pa.ParsedField(
            description="q2", parser=system_one.Noul(), source="b"
        )

    with system_one.provider(_Fixed(value=0.9)):
        out = Model.validate(pd.DataFrame({"a": ["x"], "b": ["y"]}))
    stats = system_one.stats(out)
    assert stats["batches"] == 2
    assert stats["called"] == 2


def test_stats_of_a_plain_frame_are_empty():
    assert system_one.stats(pd.DataFrame({"a": [1]})) == {}


# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------


def test_questions_compiles_without_a_provider():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(
                description="Which team should handle this",
                parser=system_one.Choice(),
            ),
        )
    )
    assert system_one.get_provider() is None
    compiled = system_one.questions(Model)
    assert set(compiled) == {"department"}
    assert compiled["department"].instructions == (
        "Which team should handle this"
    )
    assert compiled["department"].options == ("billing", "technical")


def test_questions_omits_confidence_columns():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        ),
        conf=(
            float,
            pa.ParsedField(parser=system_one.Confidence("department")),
        ),
    )
    assert set(system_one.questions(Model)) == {"department"}


def test_questions_accepts_a_schema_too():
    schema = pa.DataFrameSchema(
        {
            "body": pa.Column(str),
            "flag": pa.ParsedColumn(
                bool,
                source="body",
                description="q",
                parser=system_one.Noul(),
            ),
        }
    )
    assert set(system_one.questions(schema)) == {"flag"}


def test_plan_reports_requests_without_sending_them():
    Model = _model(
        department=(
            Department,
            pa.ParsedField(description="q", parser=system_one.Choice()),
        ),
        urgent=(
            bool,
            pa.ParsedField(description="q2", parser=system_one.Noul()),
        ),
    )
    frame = pd.DataFrame({"body": ["a ticket body", "another one"]})
    plan = system_one.plan(Model, frame)

    assert plan.rows == 2
    assert plan.batches == 1
    assert plan.questions == 2
    # two columns, but one request per row
    assert plan.requests == 2
    assert plan.estimated_input_tokens > 0


def test_plan_counts_batches_separately():
    class Model(pa.DataFrameModel):
        a: str
        b: str
        from_a: bool = pa.ParsedField(
            description="q1", parser=system_one.Noul(), source="a"
        )
        from_b: bool = pa.ParsedField(
            description="q2", parser=system_one.Noul(), source="b"
        )

    plan = system_one.plan(Model, pd.DataFrame({"a": ["x"], "b": ["y"]}))
    assert plan.batches == 2
    assert plan.requests == 2


def test_plan_without_data():
    Model = _model(
        urgent=(
            bool,
            pa.ParsedField(description="q", parser=system_one.Noul()),
        )
    )
    plan = system_one.plan(Model, rows=1000)
    assert plan.rows == 1000
    assert plan.requests == 1000
    assert plan.estimated_input_tokens == 0


def test_plan_on_a_schema_with_no_system_one_columns():
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    plan = system_one.plan(schema, pd.DataFrame({"a": [1]}))
    assert plan.batches == 0
    assert plan.requests == 0


def _priced(price):
    return system_one.MockProvider(
        capabilities=system_one.ProviderCapabilities(
            price_per_million_input_tokens=price
        )
    )


def _one_noul_model():
    return _model(
        flag=(bool, pa.ParsedField(description="q", parser=system_one.Noul()))
    )


def test_plan_prices_the_input_when_the_provider_declares_a_price():
    frame = pd.DataFrame({"body": ["a ticket body", "another one"]})
    plan = system_one.plan(_one_noul_model(), frame, provider=_priced(2.0))
    assert plan.estimated_cost_usd == pytest.approx(
        plan.estimated_input_tokens * 2.0 / 1e6
    )


def test_plan_cost_is_unknown_not_zero_without_a_price():
    frame = pd.DataFrame({"body": ["a ticket body"]})
    assert (
        system_one.plan(
            _one_noul_model(), frame, provider=_priced(None)
        ).estimated_cost_usd
        is None
    )
    # no provider at all: nothing to price with
    assert system_one.plan(_one_noul_model(), frame).estimated_cost_usd is None


def test_a_local_model_is_free_not_unknown():
    frame = pd.DataFrame({"body": ["a ticket body"]})
    plan = system_one.plan(
        _one_noul_model(), frame, provider=system_one.OllayaProvider("laya")
    )
    assert plan.estimated_cost_usd == 0.0


def test_plan_without_data_has_no_cost_rather_than_a_zero():
    plan = system_one.plan(_one_noul_model(), rows=100, provider=_priced(2.0))
    assert plan.estimated_cost_usd is None


def test_questions_can_be_checked_against_a_provider_without_a_request():
    class Big(str, enum.Enum):
        a = "a"
        b = "b"
        c = "c"

    Model = _model(
        big=(Big, pa.ParsedField(description="q", parser=system_one.Choice()))
    )
    narrow = system_one.MockProvider(
        capabilities=system_one.ProviderCapabilities(max_options=2)
    )
    assert set(system_one.questions(Model)) == {"big"}  # fine with no provider
    with pytest.raises(ProviderCapabilityError, match="3 options"):
        system_one.questions(Model, provider=narrow)
    assert narrow.calls == 0
