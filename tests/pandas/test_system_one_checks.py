"""Tests for ``system_one.Holds`` semantic checks."""

import pandas as pd
import pytest

import pandera.pandas as pa
import pandera.system_one as system_one
from pandera.errors import SchemaError, SchemaErrors
from pandera.system_one import primitives as q
from pandera.system_one.providers.base import ENABLED_ENV_VAR


class _Verdict:
    """Answers every noul with a fixed probability, recording the states."""

    id = "verdict"
    model_version = "verdict-1"

    def __init__(self, probability=1.0, per_state=None):
        self.probability = probability
        self.per_state = per_state or {}
        self.states = []

    @property
    def limits(self):
        return system_one.ProviderLimits(max_concurrency=4)

    def compile(self, questions):
        self.questions = dict(questions)
        return dict(questions)

    async def decide(self, state, compiled):
        self.states.append(state)
        key = state if isinstance(state, str) else None
        probability = self.per_state.get(key, self.probability)
        return {name: q.Decision(value=probability) for name in compiled}


def test_column_check_judges_the_columns_values():
    schema = pa.DataFrameSchema(
        {
            "description": pa.Column(
                str,
                checks=system_one.Holds("The text is coherent"),
            )
        }
    )
    provider = _Verdict(probability=1.0)
    with system_one.provider(provider):
        schema.validate(pd.DataFrame({"description": ["a", "b"]}))
    assert provider.states == ["a", "b"]


def test_failing_check_reports_failure_cases():
    schema = pa.DataFrameSchema(
        {
            "description": pa.Column(
                str,
                checks=system_one.Holds(
                    "The text is coherent", min_probability=0.9
                ),
            )
        }
    )
    with system_one.provider(_Verdict(probability=0.1)):
        with pytest.raises(SchemaError) as excinfo:
            schema.validate(pd.DataFrame({"description": ["nonsense"]}))
    assert "nonsense" in str(excinfo.value)


def test_min_probability_is_the_threshold():
    def _schema(threshold):
        return pa.DataFrameSchema(
            {
                "d": pa.Column(
                    str,
                    checks=system_one.Holds("q", min_probability=threshold),
                )
            }
        )

    frame = pd.DataFrame({"d": ["x"]})
    with system_one.provider(_Verdict(probability=0.7)):
        _schema(0.6).validate(frame)  # passes
        with pytest.raises((SchemaError, SchemaErrors)):
            _schema(0.8).validate(frame)


def test_dataframe_check_sends_the_context_columns():
    schema = pa.DataFrameSchema(
        {"name": pa.Column(str), "category": pa.Column(str)},
        checks=system_one.Holds(
            "The name fits the category", context=["name", "category"]
        ),
    )
    provider = _Verdict()
    with system_one.provider(provider):
        schema.validate(
            pd.DataFrame({"name": ["hammer"], "category": ["tools"]})
        )
    assert provider.states == [{"name": "hammer", "category": "tools"}]


def test_context_on_a_column_check_explains_the_fix():
    schema = pa.DataFrameSchema(
        {"d": pa.Column(str, checks=system_one.Holds("q", context=["a"]))}
    )
    with system_one.provider(_Verdict()):
        with pytest.raises(SchemaError, match="dataframe-level check"):
            schema.validate(pd.DataFrame({"d": ["x"], "a": ["y"]}))


def test_dataframe_check_without_context_explains_the_fix():
    schema = pa.DataFrameSchema(
        {"d": pa.Column(str)}, checks=system_one.Holds("q")
    )
    with system_one.provider(_Verdict()):
        with pytest.raises(SchemaError, match="nothing specific to judge"):
            schema.validate(pd.DataFrame({"d": ["x"]}))


def test_unknown_context_column():
    schema = pa.DataFrameSchema(
        {"d": pa.Column(str)},
        checks=system_one.Holds("q", context=["nope"]),
    )
    with system_one.provider(_Verdict()):
        with pytest.raises(SchemaError, match="not in the dataframe"):
            schema.validate(pd.DataFrame({"d": ["x"]}))


def test_lazy_validation_collects_semantic_failures():
    schema = pa.DataFrameSchema(
        {
            "d": pa.Column(
                str, checks=system_one.Holds("q", min_probability=0.9)
            ),
            "n": pa.Column(int),
        }
    )
    with system_one.provider(_Verdict(probability=0.0)):
        with pytest.raises(SchemaErrors) as excinfo:
            schema.validate(
                pd.DataFrame({"d": ["a"], "n": ["not an int"]}), lazy=True
            )
    # both the semantic failure and the dtype failure are reported
    assert len(excinfo.value.schema_errors) >= 2


def test_per_row_verdicts():
    schema = pa.DataFrameSchema(
        {
            "d": pa.Column(
                str,
                checks=system_one.Holds("q", min_probability=0.5),
            )
        }
    )
    provider = _Verdict(per_state={"good": 0.9, "bad": 0.1})
    with system_one.provider(provider):
        with pytest.raises(SchemaError) as excinfo:
            schema.validate(pd.DataFrame({"d": ["good", "bad"]}))
    failure_cases = excinfo.value.failure_cases["failure_case"].tolist()
    assert failure_cases == ["bad"]


def test_check_is_skipped_when_disabled(monkeypatch):
    """A schema carrying semantic checks still has to run offline."""
    monkeypatch.setenv(ENABLED_ENV_VAR, "0")
    schema = pa.DataFrameSchema(
        {
            "d": pa.Column(
                str, checks=system_one.Holds("q", min_probability=0.9)
            )
        }
    )
    # no provider configured, and a probability that would fail anyway
    with pytest.warns(UserWarning, match="disabled"):
        schema.validate(pd.DataFrame({"d": ["x"]}))


def test_no_provider_raises():
    schema = pa.DataFrameSchema(
        {"d": pa.Column(str, checks=system_one.Holds("q"))}
    )
    assert system_one.get_provider() is None
    with pytest.raises(SchemaError, match="no provider is configured"):
        schema.validate(pd.DataFrame({"d": ["x"]}))


def test_explicit_provider_overrides_the_configured_one():
    explicit = _Verdict(probability=1.0)
    schema = pa.DataFrameSchema(
        {"d": pa.Column(str, checks=system_one.Holds("q", provider=explicit))}
    )
    # the ambient provider would fail the check; the explicit one passes it
    with system_one.provider(_Verdict(probability=0.0)):
        schema.validate(pd.DataFrame({"d": ["x"]}))
    assert explicit.states == ["x"]


def test_the_instructions_become_the_check_description():
    check = system_one.Holds("The text is coherent")
    assert check.description == "The text is coherent"


def test_model_dataframe_check_composition():
    class Products(pa.DataFrameModel):
        name: str
        category: str

        @pa.dataframe_check
        def name_fits_category(cls, df):
            return system_one.holds(
                "The name fits the category",
                context=["name", "category"],
                min_probability=0.5,
            )(df)

    with system_one.provider(_Verdict(probability=0.9)):
        Products.validate(
            pd.DataFrame({"name": ["hammer"], "category": ["tools"]})
        )

    with system_one.provider(_Verdict(probability=0.1)):
        with pytest.raises((SchemaError, SchemaErrors)):
            Products.validate(
                pd.DataFrame({"name": ["hammer"], "category": ["produce"]})
            )
