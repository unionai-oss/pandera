"""Tests for parsers that declare the columns they read and produce."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.api.parsers import Parser, order_parsers
from pandera.errors import (
    ParserSourceError,
    ParserTargetError,
    SchemaError,
    SchemaErrors,
    SchemaInitError,
)

# --------------------------------------------------------------------------
# declaration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a", ("a",)),
        (["a"], ("a",)),
        (["a", "b"], ("a", "b")),
        (("a", "b"), ("a", "b")),
        (None, None),
    ],
)
def test_source_target_normalized_to_tuple(value, expected):
    parser = Parser(lambda s: s, source=value, target=value)
    assert parser.source == expected
    assert parser.target == expected


def test_derives_columns_flag():
    assert not Parser(lambda s: s).derives_columns
    assert Parser(lambda s: s, source="a").derives_columns
    assert Parser(lambda s: s, target="a").derives_columns


@pytest.mark.parametrize("argument", ["source", "target"])
def test_empty_declaration_raises(argument):
    with pytest.raises(SchemaInitError, match=f"empty {argument}"):
        Parser(lambda s: s, **{argument: []})


@pytest.mark.parametrize("argument", ["source", "target"])
def test_non_string_declaration_raises(argument):
    with pytest.raises(SchemaInitError, match="must be column names"):
        Parser(lambda s: s, **{argument: [1]})


@pytest.mark.parametrize("argument", ["source", "target"])
def test_duplicate_declaration_raises(argument):
    with pytest.raises(SchemaInitError, match="duplicate columns"):
        Parser(lambda s: s, **{argument: ["a", "a"]})


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


def test_single_source_single_target_is_series_to_series():
    schema = pa.DataFrameSchema(
        {"body": pa.Column(str), "n_words": pa.Column(int)},
        parsers=[
            Parser(
                lambda s: s.str.split().str.len(),
                source="body",
                target="n_words",
            )
        ],
    )
    out = schema.validate(pd.DataFrame({"body": ["a b c", "d e"]}))
    assert out["n_words"].tolist() == [3, 2]


def test_multiple_sources_receive_a_dataframe():
    def _combine(df):
        return df["first"] + " " + df["last"]

    schema = pa.DataFrameSchema(
        {
            "first": pa.Column(str),
            "last": pa.Column(str),
            "full": pa.Column(str),
        },
        parsers=[Parser(_combine, source=["first", "last"], target="full")],
    )
    out = schema.validate(
        pd.DataFrame({"first": ["ada"], "last": ["lovelace"]})
    )
    assert out["full"].tolist() == ["ada lovelace"]


def test_multiple_targets_from_a_dataframe():
    def _split(df):
        parts = df["name"].str.split(" ", expand=True)
        return df.assign(first=parts[0], last=parts[1])

    schema = pa.DataFrameSchema(
        {
            "name": pa.Column(str),
            "first": pa.Column(str),
            "last": pa.Column(str),
        },
        parsers=[Parser(_split, source=["name"], target=["first", "last"])],
    )
    out = schema.validate(pd.DataFrame({"name": ["ada lovelace"]}))
    assert out["first"].tolist() == ["ada"]
    assert out["last"].tolist() == ["lovelace"]


def test_element_wise_derivation():
    schema = pa.DataFrameSchema(
        {"body": pa.Column(str), "n_chars": pa.Column(int)},
        parsers=[
            Parser(len, source="body", target="n_chars", element_wise=True)
        ],
    )
    out = schema.validate(pd.DataFrame({"body": ["abc", "de"]}))
    assert out["n_chars"].tolist() == [3, 2]


def test_derived_column_is_coerced_and_checked():
    """A derived column is an ordinary column downstream: it gets coerced and
    its checks run, which is the whole point of declaring it in the schema."""
    schema = pa.DataFrameSchema(
        {
            "body": pa.Column(str),
            "n_words": pa.Column(int, pa.Check.ge(1), coerce=True),
        },
        parsers=[
            Parser(
                lambda s: s.str.split().str.len().astype(str),
                source="body",
                target="n_words",
            )
        ],
    )
    out = schema.validate(pd.DataFrame({"body": ["a b c"]}))
    assert out["n_words"].tolist() == [3]

    failing = pa.DataFrameSchema(
        {"body": pa.Column(str), "n_words": pa.Column(int, pa.Check.ge(5))},
        parsers=[
            Parser(
                lambda s: s.str.split().str.len(),
                source="body",
                target="n_words",
            )
        ],
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        failing.validate(pd.DataFrame({"body": ["a b c"]}))


def test_derived_column_allowed_under_strict():
    schema = pa.DataFrameSchema(
        {"body": pa.Column(str), "n": pa.Column(int)},
        parsers=[Parser(lambda s: s.str.len(), source="body", target="n")],
        strict=True,
    )
    assert schema.validate(pd.DataFrame({"body": ["abc"]}))["n"].tolist() == [
        3
    ]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_missing_source_raises_parser_source_error():
    """Regression: this used to surface as a bare ``KeyError: 'nope'`` with no
    schema context and no indication of which parser failed."""
    schema = pa.DataFrameSchema(
        {"n": pa.Column(int)},
        parsers=[
            Parser(lambda s: s, source="nope", target="n", name="derive_n")
        ],
    )
    with pytest.raises(ParserSourceError) as excinfo:
        schema.validate(pd.DataFrame({"body": ["a"]}))

    message = str(excinfo.value)
    assert "derive_n" in message  # which parser
    assert "nope" in message  # which column it wanted
    assert "body" in message  # what was actually there


def test_missing_target_raises_parser_target_error():
    schema = pa.DataFrameSchema(
        {
            "body": pa.Column(str),
            "p": pa.Column(int),
            "q": pa.Column(int),
        },
        parsers=[
            Parser(
                lambda df: df.assign(p=1),
                source=["body"],
                target=["p", "q"],
                name="derive_pq",
            )
        ],
    )
    with pytest.raises(ParserTargetError) as excinfo:
        schema.validate(pd.DataFrame({"body": ["a"]}))

    message = str(excinfo.value)
    assert "derive_pq" in message
    assert "q" in message


def test_multi_target_parser_must_return_dataframe():
    schema = pa.DataFrameSchema(
        {"body": pa.Column(str), "p": pa.Column(int), "q": pa.Column(int)},
        parsers=[
            Parser(
                lambda df: df["body"],
                source=["body"],
                target=["p", "q"],
            )
        ],
    )
    with pytest.raises(ParserTargetError, match="must return a DataFrame"):
        schema.validate(pd.DataFrame({"body": ["a"]}))


def test_source_only_parser_must_return_dataframe():
    schema = pa.DataFrameSchema(
        {"body": pa.Column(str)},
        parsers=[Parser(lambda df: df["body"], source=["body"])],
    )
    with pytest.raises(ParserTargetError, match="must return a DataFrame"):
        schema.validate(pd.DataFrame({"body": ["a"]}))


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------


def test_parsers_run_in_dependency_order_not_list_order():
    schema = pa.DataFrameSchema(
        {"body": pa.Column(str), "a": pa.Column(int), "b": pa.Column(int)},
        parsers=[
            # declared first, but depends on `a`
            Parser(lambda s: s * 2, source="a", target="b"),
            Parser(lambda s: s.str.len(), source="body", target="a"),
        ],
    )
    out = schema.validate(pd.DataFrame({"body": ["abc"]}))
    assert out["a"].tolist() == [3]
    assert out["b"].tolist() == [6]


def test_cycle_raises():
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int), "y": pa.Column(int)},
        parsers=[
            Parser(lambda s: s, source="x", target="y"),
            Parser(lambda s: s, source="y", target="x"),
        ],
    )
    with pytest.raises(SchemaInitError, match="cycle"):
        schema.validate(pd.DataFrame({"x": [1]}))


def test_self_dependency_is_not_a_cycle():
    """A parser that reads and writes the same column is a transform, not a
    cycle -- it depends on the data, not on its own output."""
    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        parsers=[Parser(lambda s: s * 2, source="a", target="a")],
    )
    assert schema.validate(pd.DataFrame({"a": [1]}))["a"].tolist() == [2]


def test_undeclared_parsers_run_first_and_keep_list_order():
    calls = []

    def _record(name):
        def _fn(df):
            calls.append(name)
            return df

        return _fn

    ordered = order_parsers(
        [
            Parser(lambda s: s, source="a", target="b", name="declared"),
            Parser(_record("first"), name="first"),
            Parser(_record("second"), name="second"),
        ]
    )
    assert [p.name for p in ordered] == ["first", "second", "declared"]


def test_order_parsers_is_stable_for_independent_parsers():
    parsers = [
        Parser(lambda s: s, source="x", target="a", name="a"),
        Parser(lambda s: s, source="x", target="b", name="b"),
        Parser(lambda s: s, source="x", target="c", name="c"),
    ]
    assert [p.name for p in order_parsers(parsers)] == ["a", "b", "c"]


def test_order_parsers_resolves_chains():
    parsers = [
        Parser(lambda s: s, source="c", target="d", name="c_to_d"),
        Parser(lambda s: s, source="b", target="c", name="b_to_c"),
        Parser(lambda s: s, source="a", target="b", name="a_to_b"),
    ]
    assert [p.name for p in order_parsers(parsers)] == [
        "a_to_b",
        "b_to_c",
        "c_to_d",
    ]


# --------------------------------------------------------------------------
# backwards compatibility
# --------------------------------------------------------------------------


def test_undeclared_parsers_behave_exactly_as_before():
    """Parsers with no source/target take the original code path."""
    column_parser = pa.DataFrameSchema(
        {"a": pa.Column(int, parsers=pa.Parser(lambda s: s.clip(lower=0)))}
    )
    assert column_parser.validate(pd.DataFrame({"a": [1, -1]}))[
        "a"
    ].tolist() == [1, 0]

    frame_parser = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        parsers=pa.Parser(lambda df: df.assign(a=df["a"] + 1)),
    )
    assert frame_parser.validate(pd.DataFrame({"a": [1]}))["a"].tolist() == [2]


def test_parser_kwargs_are_still_forwarded_to_the_function():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(
                int, parsers=pa.Parser(lambda s, offset: s + offset, offset=10)
            )
        }
    )
    assert schema.validate(pd.DataFrame({"a": [1]}))["a"].tolist() == [11]


def test_source_and_target_are_no_longer_forwarded_as_kwargs():
    """``Parser`` forwards unrecognized keyword arguments to the parser
    function, so ``source``/``target`` used to arrive as function kwargs.
    Promoting them to real parameters is a (small) breaking change for anyone
    whose parser function took a keyword by those names.
    """
    received = {}

    def _fn(series, **kwargs):
        received.update(kwargs)
        return series

    schema = pa.DataFrameSchema(
        {"a": pa.Column(int)},
        parsers=[Parser(_fn, source="a", target="a")],
    )
    schema.validate(pd.DataFrame({"a": [1]}))
    assert received == {}
