"""Preserve column row-dropping configuration through schema transforms."""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.errors import SchemaErrors


@pytest.mark.parametrize("drop_invalid_rows", [False, True])
def test_column_properties_roundtrip_drop_invalid_rows(drop_invalid_rows):
    column = pa.Column(
        int, pa.Check.gt(0), name="x", drop_invalid_rows=drop_invalid_rows
    )
    assert column.properties["drop_invalid_rows"] is drop_invalid_rows
    assert pa.Column(**column.properties) == column


@pytest.mark.parametrize("drop_invalid_rows", [False, True])
@pytest.mark.parametrize("updated", ["x", "y", None])
def test_update_columns_preserves_drop_invalid_rows(
    drop_invalid_rows, updated
):
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int, pa.Check.gt(0), drop_invalid_rows=drop_invalid_rows
            ),
            "y": pa.Column(int),
        }
    )
    updates = {} if updated is None else {updated: {"nullable": True}}
    result = schema.update_columns(updates)
    assert result.columns["x"].drop_invalid_rows is drop_invalid_rows
    assert result.columns["y"].drop_invalid_rows is False
    assert schema.columns["x"].drop_invalid_rows is drop_invalid_rows
    assert result.columns["x"] is not schema.columns["x"]


@pytest.mark.parametrize("updated", ["x", "y", None])
def test_updated_column_still_drops_invalid_rows(updated):
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(int, pa.Check.gt(0), drop_invalid_rows=True),
            "y": pa.Column(int),
        }
    )
    updates = {} if updated is None else {updated: {"nullable": True}}
    result = schema.update_columns(updates)
    data = pd.DataFrame({"x": [-1, 1, 2], "y": [10, 20, 30]})
    before = data.copy(deep=True)
    actual = result.columns["x"].validate(data, lazy=True)
    pd.testing.assert_frame_equal(actual, data.iloc[1:])
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("drop_invalid_rows", [False, True])
def test_update_columns_can_explicitly_override_drop_invalid_rows(
    drop_invalid_rows,
):
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int, pa.Check.gt(0), drop_invalid_rows=not drop_invalid_rows
            )
        }
    )
    updated = schema.update_columns(
        {"x": {"drop_invalid_rows": drop_invalid_rows}}
    )
    data = pd.DataFrame({"x": [-1, 1, 2]})
    if drop_invalid_rows:
        actual = updated.columns["x"].validate(data, lazy=True)
        pd.testing.assert_frame_equal(actual, data.iloc[1:])
    else:
        with pytest.raises(SchemaErrors):
            updated.columns["x"].validate(data, lazy=True)
    assert schema.columns["x"].drop_invalid_rows is not drop_invalid_rows
