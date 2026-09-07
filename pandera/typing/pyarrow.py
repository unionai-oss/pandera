"""Pandera type annotations for PyArrow."""

import functools
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from packaging import version

from pandera.typing.common import DataFrameBase, DataFrameModel
from pandera.typing.formats import Formats

try:
    import pyarrow

    PYARROW_INSTALLED = True
except ImportError:
    PYARROW_INSTALLED = False


def pyarrow_version():
    """Return the pyarrow version."""
    return version.parse(pyarrow.__version__)


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if PYARROW_INSTALLED:

    class Table(DataFrameBase, pyarrow.Table, Generic[T]):
        """Pandera generic for ``pyarrow.Table``, only used for annotation."""

        @classmethod
        def from_format(cls, obj: Any, config) -> "pyarrow.Table":
            """Convert serialized data into a ``pyarrow.Table``.

            The format is taken from the
            :py:class:`pandera.api.pyarrow.model.DataFrameModel` config
            options ``from_format`` and ``from_format_kwargs``.

            :param obj: object representing a serialized table.
            :param config: dataframe model configuration object.
            """
            if config.from_format is None:
                if not isinstance(obj, pyarrow.Table):
                    try:
                        obj = pyarrow.table(obj)
                    except Exception as exc:
                        raise ValueError(
                            f"Expected pyarrow.Table, found {type(obj)}"
                        ) from exc
                return obj

            if callable(config.from_format):
                reader = config.from_format
                return reader(obj, **(config.from_format_kwargs or {}))

            try:
                format_type = Formats(config.from_format)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported format: {config.from_format}. "
                    "PyArrow natively supports: dict, csv, json, and parquet."
                ) from exc

            kwargs = config.from_format_kwargs or {}

            if format_type == Formats.dict:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected dict for dict format, got {type(obj)}"
                    )
                return pyarrow.table(obj)

            try:
                if format_type == Formats.csv:
                    from pyarrow import csv

                    return csv.read_csv(obj, **kwargs)
                if format_type == Formats.json:
                    from pyarrow import json as pa_json

                    return pa_json.read_json(obj, **kwargs)
                if format_type == Formats.parquet:
                    from pyarrow import parquet

                    return parquet.read_table(obj, **kwargs)
                if format_type == Formats.feather:
                    from pyarrow import feather

                    return feather.read_table(obj, **kwargs)
            except Exception as exc:
                raise ValueError(
                    f"Failed to read {format_type.value} with PyArrow: {exc}"
                ) from exc

            raise ValueError(
                f"{format_type.value} format is not natively supported by "
                "PyArrow. Use a custom callable for from_format instead."
            )

        @classmethod
        def to_format(cls, data: "pyarrow.Table", config) -> Any:
            """Convert a table to the format specified in the model config.

            Driven by the
            :py:class:`pandera.api.pyarrow.model.DataFrameModel` config
            options ``to_format`` and ``to_format_kwargs``.

            :param data: convert this data to the specified format
            :param config: config object from the DataFrameModel
            """
            if config.to_format is None:
                return data

            if callable(config.to_format):
                writer = functools.partial(config.to_format, data)
                buffer = (
                    config.to_format_buffer()
                    if callable(config.to_format_buffer)
                    else None
                )
                args = [] if buffer is None else [buffer]
                out = writer(*args, **(config.to_format_kwargs or {}))
                return out if buffer is None else buffer

            try:
                format_type = Formats(config.to_format)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported format: {config.to_format}. "
                    "PyArrow natively supports: dict, parquet, and feather."
                ) from exc

            kwargs = config.to_format_kwargs or {}

            if format_type == Formats.dict:
                return data.to_pydict()
            if format_type == Formats.parquet:
                from pyarrow import parquet

                return parquet.write_table(data, **kwargs)
            if format_type == Formats.feather:
                from pyarrow import feather

                return feather.write_feather(data, **kwargs)

            raise ValueError(
                f"{format_type.value} format is not natively supported by "
                "PyArrow. Use a custom callable for to_format instead."
            )

        @classmethod
        def _get_schema_model(cls, field):
            if not field.sub_fields:
                raise TypeError(
                    "Expected a typed pandera.typing.pyarrow.Table,"
                    " e.g. Table[Schema]"
                )
            return field.sub_fields[0].type_
