"""Tests for :mod:`pandera.io.polars_io`."""

import polars as pl
import pytest

import pandera.polars as pa
from pandera.errors import SchemaDefinitionError, SchemaError


def test_polars_yaml_json_roundtrip():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, nullable=False),
            "b": pa.Column(str),
        },
        strict=True,
    )
    from pandera.io import polars_io

    y = polars_io.to_yaml(schema, minimal=True)
    assert "polars_dataframe" in y
    loaded = polars_io.from_yaml(y)
    assert isinstance(loaded, pa.DataFrameSchema)
    assert loaded == schema

    j = polars_io.to_json(schema, minimal=True)
    assert "polars_dataframe" in j
    loaded_j = polars_io.from_json(j)
    assert isinstance(loaded_j, pa.DataFrameSchema)
    assert loaded_j == schema


def test_polars_full_serdes_includes_version():
    """``minimal=False`` retains package version in the payload."""
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    from pandera.io import polars_io

    payload = polars_io.serialize_schema(schema, minimal=False)
    assert "version" in payload
    assert payload["api"] == "polars"
    assert (
        polars_io.from_yaml(polars_io.to_yaml(schema, minimal=False)) == schema
    )


def test_polars_rejects_pandas_schema_type():
    from pandera.io import polars_io

    bad = {"schema_type": "dataframe", "columns": {}}
    with pytest.raises(SchemaDefinitionError):
        polars_io.deserialize_schema(bad)


@pytest.mark.parametrize(
    "check_kwargs, expected_name, expected_error",
    [
        [{}, "greater_than", "greater_than(0)"],
        [{"name": "positive"}, "positive", "greater_than(0)"],
        [{"error": "boom"}, "greater_than", "boom"],
        [{"name": "positive", "error": "boom"}, "positive", "boom"],
    ],
)
def test_polars_check_options_survive_yaml_roundtrip(
    check_kwargs,
    expected_name,
    expected_error,
):
    """Flat check payloads must not read options back as check parameters."""
    from pandera.io import polars_io

    schema = pa.DataFrameSchema(
        {"a": pa.Column(int, checks=[pa.Check.gt(0, **check_kwargs)])}
    )
    loaded = polars_io.from_yaml(polars_io.to_yaml(schema, minimal=False))

    assert loaded == schema
    (check,) = loaded.columns["a"].checks
    assert check.name == expected_name
    assert check.error == expected_error
    with pytest.raises(SchemaError):
        loaded.validate(pl.DataFrame({"a": [-1]}))


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Categorical,
        pl.Enum(["a", "b"]),
        pl.Datetime("us", "UTC"),
        pl.Duration("ms"),
        pl.Decimal(10, 2),
        pl.List(pl.Int64),
        pl.Array(pl.Int64, 3),
        pl.Struct({"a": pl.Int64, "b": pl.List(pl.String)}),
        pl.List(pl.Struct({"a": pl.Enum(["x", "y"])})),
    ],
    ids=str,
)
@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_polars_roundtrip_parametrized_dtypes(dtype, fmt):
    """Parametrized polars dtypes are read back with their parameters."""
    schema = pa.DataFrameSchema({"a": pa.Column(dtype)})

    serialized = getattr(schema, f"to_{fmt}")()
    restored = getattr(pa.DataFrameSchema, f"from_{fmt}")(serialized)

    restored_dtype = restored.columns["a"].dtype.type
    assert type(restored_dtype) is type(schema.columns["a"].dtype.type)
    assert str(restored_dtype) == str(schema.columns["a"].dtype.type)


@pytest.mark.parametrize(
    "text",
    ["__import__('os').getcwd()", "Int64.__class__", "NotADtype"],
)
def test_polars_from_yaml_rejects_non_dtype_text(text):
    """Text that isn't a polars dtype fails as before and is never run."""
    yaml_text = (
        f'schema_type: polars_dataframe\ncolumns:\n  a:\n    dtype: "{text}"\n'
    )
    with pytest.raises(TypeError, match="cannot parse input"):
        pa.DataFrameSchema.from_yaml(yaml_text)
