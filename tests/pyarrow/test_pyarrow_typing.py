"""Tests for the pyarrow typing module."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow
import pytest
from pyarrow import feather, parquet

from pandera.typing.pyarrow import Table, pyarrow_version


def _config(**kwargs):
    """Build a stand-in for a DataFrameModel config object."""
    defaults = {
        "from_format": None,
        "from_format_kwargs": None,
        "to_format": None,
        "to_format_kwargs": None,
        "to_format_buffer": None,
    }
    return SimpleNamespace(**{**defaults, **kwargs})


TABLE = pyarrow.table({"a": [1, 2], "b": ["x", "y"]})


def test_pyarrow_version():
    assert str(pyarrow_version()) == pyarrow.__version__


class TestFromFormat:
    """Test Table.from_format."""

    def test_none(self):
        assert Table.from_format(TABLE, _config()) is TABLE

    def test_none_converts_mapping(self):
        result = Table.from_format({"a": [1, 2], "b": ["x", "y"]}, _config())
        assert result.equals(TABLE)

    def test_none_rejects_unconvertible(self):
        with pytest.raises(ValueError, match="Expected pyarrow.Table"):
            Table.from_format(1, _config())

    def test_callable(self):
        reader = MagicMock(return_value=TABLE)
        config = _config(from_format=reader, from_format_kwargs={"n": 1})

        assert Table.from_format("data", config) is TABLE
        reader.assert_called_once_with("data", n=1)

    def test_dict(self):
        config = _config(from_format="dict")
        result = Table.from_format({"a": [1, 2], "b": ["x", "y"]}, config)
        assert result.equals(TABLE)

    def test_dict_rejects_non_dict(self):
        with pytest.raises(ValueError, match="Expected dict for dict format"):
            Table.from_format(TABLE, _config(from_format="dict"))

    def test_csv(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("a,b\n1,x\n2,y\n")

        result = Table.from_format(str(path), _config(from_format="csv"))
        assert result.equals(TABLE)

    def test_json(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

        result = Table.from_format(str(path), _config(from_format="json"))
        assert result.equals(TABLE)

    def test_parquet(self, tmp_path):
        path = tmp_path / "t.parquet"
        parquet.write_table(TABLE, where=str(path))

        result = Table.from_format(str(path), _config(from_format="parquet"))
        assert result.equals(TABLE)

    def test_feather(self, tmp_path):
        path = tmp_path / "t.feather"
        feather.write_feather(TABLE, dest=str(path))

        result = Table.from_format(str(path), _config(from_format="feather"))
        assert result.equals(TABLE)

    def test_kwargs_are_forwarded(self, tmp_path):
        path = tmp_path / "t.parquet"
        parquet.write_table(TABLE, where=str(path))

        config = _config(
            from_format="parquet", from_format_kwargs={"columns": ["a"]}
        )
        assert Table.from_format(str(path), config).column_names == ["a"]

    def test_read_failure_is_wrapped(self):
        with patch("pyarrow.csv.read_csv", side_effect=OSError("boom")):
            with pytest.raises(
                ValueError, match="Failed to read csv with PyArrow: boom"
            ):
                Table.from_format("t.csv", _config(from_format="csv"))

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported format: nope"):
            Table.from_format("data", _config(from_format="nope"))

    @pytest.mark.parametrize("fmt", ["pickle", "json_normalize"])
    def test_format_not_supported_by_pyarrow(self, fmt):
        with pytest.raises(
            ValueError, match=f"{fmt} format is not natively supported"
        ):
            Table.from_format("data", _config(from_format=fmt))


class TestToFormat:
    """Test Table.to_format."""

    def test_none(self):
        assert Table.to_format(TABLE, _config()) is TABLE

    def test_callable_without_buffer(self):
        writer = MagicMock(return_value="written")
        config = _config(to_format=writer, to_format_kwargs={"n": 1})

        assert Table.to_format(TABLE, config) == "written"
        writer.assert_called_once_with(TABLE, n=1)

    def test_callable_with_buffer(self):
        buffer = io.BytesIO()
        writer = MagicMock(return_value=None)
        config = _config(
            to_format=writer,
            to_format_kwargs={"n": 1},
            to_format_buffer=lambda: buffer,
        )

        assert Table.to_format(TABLE, config) is buffer
        writer.assert_called_once_with(TABLE, buffer, n=1)

    def test_dict(self):
        result = Table.to_format(TABLE, _config(to_format="dict"))
        assert result == {"a": [1, 2], "b": ["x", "y"]}

    def test_parquet(self, tmp_path):
        path = tmp_path / "t.parquet"
        config = _config(
            to_format="parquet", to_format_kwargs={"where": str(path)}
        )

        Table.to_format(TABLE, config)
        assert parquet.read_table(str(path)).equals(TABLE)

    def test_feather(self, tmp_path):
        path = tmp_path / "t.feather"
        config = _config(
            to_format="feather", to_format_kwargs={"dest": str(path)}
        )

        Table.to_format(TABLE, config)
        assert feather.read_table(str(path)).equals(TABLE)

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported format: nope"):
            Table.to_format(TABLE, _config(to_format="nope"))

    @pytest.mark.parametrize(
        "fmt", ["csv", "json", "pickle", "json_normalize"]
    )
    def test_format_not_supported_by_pyarrow(self, fmt):
        with pytest.raises(
            ValueError, match=f"{fmt} format is not natively supported"
        ):
            Table.to_format(TABLE, _config(to_format=fmt))


def test_get_schema_model():
    field = SimpleNamespace(sub_fields=[SimpleNamespace(type_="a-model")])
    assert Table._get_schema_model(field) == "a-model"


def test_get_schema_model_requires_subscript():
    with pytest.raises(TypeError, match="Expected a typed pandera.typing"):
        Table._get_schema_model(SimpleNamespace(sub_fields=None))
