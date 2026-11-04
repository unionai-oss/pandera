"""List-valued defaults are values of individual cells, not row vectors."""

import copy

import pandas as pd
import pytest

import pandera.pandas as pa


@pytest.mark.parametrize("default", [[], [1.0], [1.0, 2.0]])
@pytest.mark.parametrize("data_kind", ["nonnull", "null", "missing"])
@pytest.mark.parametrize("coerce", [False, True])
def test_list_defaults_in_dataframe(default, data_kind, coerce):
    data = {"id": [1, 2]}
    if data_kind != "missing":
        data["items"] = (
            [[2.0], [3.0]] if data_kind == "nonnull" else [None, [3.0]]
        )
    original = pd.DataFrame(data, index=[10, 20])
    before = original.copy(deep=True)
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int),
            "items": pa.Column(list[float], default=default),
        },
        add_missing_columns=True,
        coerce=coerce,
    )

    result = schema.validate(original)

    expected = (
        [[2.0], [3.0]]
        if data_kind == "nonnull"
        else [default, [3.0]]
        if data_kind == "null"
        else [default, default]
    )
    assert result["items"].tolist() == expected
    assert result.index.tolist() == [10, 20]
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("entrypoint", ["series", "column", "regex_column"])
@pytest.mark.parametrize("default", [[], [1.0, 2.0]])
def test_list_defaults_through_component_entrypoints(entrypoint, default):
    data = pd.Series([None, [3.0]], index=[7, 7], name="items_one")
    if entrypoint == "series":
        result = pa.SeriesSchema(list[float], default=default).validate(data)
    elif entrypoint == "column":
        result = pa.Column(
            list[float], name="items_one", default=default
        ).validate(data.to_frame())["items_one"]
    else:
        result = pa.DataFrameSchema(
            {r"^items_": pa.Column(list[float], default=default, regex=True)}
        ).validate(data.to_frame())["items_one"]

    assert result.tolist() == [default, [3.0]]
    assert result.index.tolist() == [7, 7]
    assert data.iloc[0] is None


@pytest.mark.parametrize("missing_column", [False, True])
def test_mutable_list_defaults_are_independent(missing_column):
    default = [{"value": [1]}]
    before = copy.deepcopy(default)
    data = pd.DataFrame({"id": [1, 2]})
    if not missing_column:
        data["items"] = None
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int),
            "items": pa.Column(list[dict], default=default),
        },
        add_missing_columns=True,
    )
    result = schema.validate(data)

    result.loc[0, "items"][0]["value"].append(2)

    assert result.loc[1, "items"] == before
    assert default == before


@pytest.mark.parametrize("missing_column", [False, True])
def test_scalar_and_index_mapping_defaults_remain_unchanged(missing_column):
    data = pd.DataFrame({"id": [1, 2]}, index=[10, 20])
    if not missing_column:
        data["scalar"] = None
        data["mapped"] = None
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int),
            "scalar": pa.Column(int, default=4),
            "mapped": pa.Column(int, default={10: 5, 20: 6}),
        },
        add_missing_columns=True,
        coerce=True,
    )

    result = schema.validate(data)

    assert result["scalar"].tolist() == [4, 4]
    assert result["mapped"].tolist() == [5, 6]


def test_series_list_default_retains_metadata():
    original = pd.Series([None, [3.0]], name="items")
    original.attrs = {"provenance": {"versions": [1]}}

    result = pa.SeriesSchema(list[float], default=[]).validate(original)

    assert result.attrs == original.attrs
    result.attrs["provenance"]["versions"].append(2)
    assert original.attrs == {"provenance": {"versions": [1]}}


@pytest.mark.parametrize("default", [[], [1.0, 2.0]])
def test_list_default_adds_empty_missing_column(default):
    data = pd.DataFrame({"id": pd.Series([], dtype=int)})
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int),
            "items": pa.Column(list[float], default=default),
        },
        add_missing_columns=True,
    )

    result = schema.validate(data)

    assert result.empty
    assert result.columns.tolist() == ["id", "items"]
    assert result["items"].dtype == object


@pytest.mark.parametrize("default", [[], [1.0, 2.0]])
def test_list_defaults_with_duplicate_column_names(default):
    data = pd.DataFrame(
        [[None, None], [None, [3.0]]], columns=["items", "items"]
    )
    schema = pa.DataFrameSchema(
        {"items": pa.Column(list[float], default=default)}
    )

    result = schema.validate(data)

    assert result.columns.tolist() == ["items", "items"]
    assert result.iloc[:, 0].tolist() == [default, default]
    assert result.iloc[:, 1].tolist() == [default, [3.0]]
    assert data.iloc[0, 0] is None
    assert data.iloc[0, 1] is None
