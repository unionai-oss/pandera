"""Tests for the DataFusion typing module."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import datafusion
import pyarrow
import pytest
from pyarrow import feather, parquet

from pandera.typing.datafusion import DataFrame, datafusion_version


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


DATA = {"a": [1, 2], "b": ["x", "y"]}
TABLE = pyarrow.table(DATA)


def _rows(df: datafusion.DataFrame) -> dict:
    return df.to_arrow_table().to_pydict()


def test_datafusion_version():
    assert str(datafusion_version()) == datafusion.__version__


class TestFromFormat:
    """Test DataFrame.from_format."""

    def test_none(self, make_df):
        df = make_df(DATA)
        assert DataFrame.from_format(df, _config()) is df

    def test_none_converts_mapping(self):
        assert _rows(DataFrame.from_format(DATA, _config())) == DATA

    def test_none_converts_table(self):
        assert _rows(DataFrame.from_format(TABLE, _config())) == DATA

    def test_none_rejects_unconvertible(self):
        with pytest.raises(ValueError, match="Expected datafusion.DataFrame"):
            DataFrame.from_format(1, _config())

    def test_callable(self, make_df):
        df = make_df(DATA)
        reader = MagicMock(return_value=df)
        config = _config(from_format=reader, from_format_kwargs={"n": 1})

        assert DataFrame.from_format("data", config) is df
        reader.assert_called_once_with("data", n=1)

    def test_dict(self):
        result = DataFrame.from_format(DATA, _config(from_format="dict"))
        assert _rows(result) == DATA

    def test_dict_rejects_non_dict(self):
        with pytest.raises(ValueError, match="Expected dict for dict format"):
            DataFrame.from_format(TABLE, _config(from_format="dict"))

    def test_csv(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("a,b\n1,x\n2,y\n")

        result = DataFrame.from_format(str(path), _config(from_format="csv"))
        assert _rows(result) == DATA

    def test_json(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

        result = DataFrame.from_format(str(path), _config(from_format="json"))
        assert _rows(result) == DATA

    def test_parquet(self, tmp_path):
        path = tmp_path / "t.parquet"
        parquet.write_table(TABLE, where=str(path))

        result = DataFrame.from_format(
            str(path), _config(from_format="parquet")
        )
        assert _rows(result) == DATA

    def test_feather(self, tmp_path):
        path = tmp_path / "t.feather"
        feather.write_feather(TABLE, str(path))

        result = DataFrame.from_format(
            str(path), _config(from_format="feather")
        )
        assert _rows(result) == DATA

    def test_kwargs_are_forwarded(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("1,x\n2,y\n")

        config = _config(
            from_format="csv", from_format_kwargs={"has_header": False}
        )
        result = DataFrame.from_format(str(path), config)
        assert list(_rows(result).values()) == [[1, 2], ["x", "y"]]

    def test_read_failure_is_wrapped(self):
        with pytest.raises(
            ValueError, match="Failed to read parquet with DataFusion"
        ):
            DataFrame.from_format(
                "/nonexistent/t.parquet", _config(from_format="parquet")
            )

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported format: nope"):
            DataFrame.from_format("data", _config(from_format="nope"))

    @pytest.mark.parametrize("fmt", ["pickle", "json_normalize"])
    def test_format_not_supported_by_datafusion(self, fmt):
        with pytest.raises(
            ValueError, match=f"{fmt} format is not natively supported"
        ):
            DataFrame.from_format("data", _config(from_format=fmt))


class TestToFormat:
    """Test DataFrame.to_format."""

    def test_none(self, make_df):
        df = make_df(DATA)
        assert DataFrame.to_format(df, _config()) is df

    def test_callable_without_buffer(self, make_df):
        df = make_df(DATA)
        writer = MagicMock(return_value="written")
        config = _config(to_format=writer, to_format_kwargs={"n": 1})

        assert DataFrame.to_format(df, config) == "written"
        writer.assert_called_once_with(df, n=1)

    def test_callable_with_buffer(self, make_df):
        df = make_df(DATA)
        buffer = io.BytesIO()
        writer = MagicMock(return_value=None)
        config = _config(
            to_format=writer,
            to_format_kwargs={"n": 1},
            to_format_buffer=lambda: buffer,
        )

        assert DataFrame.to_format(df, config) is buffer
        writer.assert_called_once_with(df, buffer, n=1)

    def test_dict(self, make_df):
        result = DataFrame.to_format(make_df(DATA), _config(to_format="dict"))
        assert result == DATA

    def test_parquet(self, make_df, tmp_path):
        path = tmp_path / "out"
        config = _config(
            to_format="parquet", to_format_kwargs={"path": str(path)}
        )

        DataFrame.to_format(make_df(DATA), config)
        assert parquet.read_table(str(path)).to_pydict() == DATA

    @pytest.mark.parametrize(
        "fmt,path_kwarg",
        [
            ("csv", "path"),
            ("json", "path"),
            ("parquet", "path"),
            ("feather", "dest"),
        ],
    )
    def test_round_trip(self, make_df, tmp_path, fmt, path_kwarg):
        """Whatever ``to_format`` writes, ``from_format`` reads back."""
        path = str(tmp_path / f"out.{fmt}")
        DataFrame.to_format(
            make_df(DATA),
            _config(to_format=fmt, to_format_kwargs={path_kwarg: path}),
        )

        result = DataFrame.from_format(path, _config(from_format=fmt))
        assert _rows(result) == DATA

    def test_csv_header_can_be_disabled(self, make_df, tmp_path):
        path = tmp_path / "out.csv"
        config = _config(
            to_format="csv",
            to_format_kwargs={"path": str(path), "with_header": False},
        )

        DataFrame.to_format(make_df(DATA), config)
        assert path.read_text().splitlines() == ["1,x", "2,y"]

    def test_unknown_format(self, make_df):
        with pytest.raises(ValueError, match="Unsupported format: nope"):
            DataFrame.to_format(make_df(DATA), _config(to_format="nope"))

    @pytest.mark.parametrize("fmt", ["pickle", "json_normalize"])
    def test_format_not_supported_by_datafusion(self, make_df, fmt):
        with pytest.raises(
            ValueError, match=f"{fmt} format is not natively supported"
        ):
            DataFrame.to_format(make_df(DATA), _config(to_format=fmt))


def test_get_schema_model():
    field = SimpleNamespace(sub_fields=[SimpleNamespace(type_="a-model")])
    assert DataFrame._get_schema_model(field) == "a-model"


def test_get_schema_model_requires_subscript():
    with pytest.raises(TypeError, match="Expected a typed pandera.typing"):
        DataFrame._get_schema_model(SimpleNamespace(sub_fields=None))
