"""Pandera type annotations for Ibis."""

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from packaging import version

from pandera.typing.common import DataFrameBase, DataFrameModel
from pandera.typing.formats import Formats

try:
    import ibis

    IBIS_INSTALLED = True
except ImportError:
    IBIS_INSTALLED = False


def ibis_version():
    """Return the Ibis version."""
    return version.parse(ibis.__version__)


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if IBIS_INSTALLED:

    class Table(DataFrameBase, ibis.Table, Generic[T]):
        """
        Pandera generic for ibis.Table, only used for type annotation.

        *new in 0.25.0*
        """

        @classmethod
        def from_format(cls, obj: Any, config) -> ibis.Table:
            """
            Converts serialized data from a specific format
            specified in the :py:class:`pandera.api.ibis.model.DataFrameModel` config options
            ``from_format`` and ``from_format_kwargs``.

            :param obj: object representing a serialized dataframe.
            :param config: dataframe model configuration object.
            """
            if config.from_format is None:
                if not isinstance(obj, ibis.Table):
                    try:
                        obj = ibis.memtable(obj)
                    except Exception as exc:
                        raise ValueError(
                            f"Expected ibis.Table, found {type(obj)}"
                        ) from exc
                return obj

            if callable(config.from_format):
                reader = config.from_format
                return reader(obj, **(config.from_format_kwargs or {}))
            else:
                # Handle different formats natively using Ibis
                try:
                    format_type = Formats(config.from_format)
                except (
                    ValueError
                ) as exc:  # pragma: no cover - tested via mocks
                    raise ValueError(
                        f"Unsupported format: {config.from_format}. "
                        f"Ibis natively supports: dict, csv, json, and parquet."
                    ) from exc

                kwargs = config.from_format_kwargs or {}

                # Define a helper function to handle format-specific reading with error handling
                def read_with_format(read_func, error_prefix):
                    """Helper to handle format-specific reading with standardized error handling.

                    Args:
                        read_func: Function to call for reading the data
                        error_prefix: Prefix for error message if the reading fails

                    Returns:
                        DataFrame: The resulting Ibis table

                    Raises:
                        ValueError: If reading fails
                    """
                    try:
                        return read_func()
                    except Exception as exc:
                        raise ValueError(f"{error_prefix}: {exc}") from exc

                # Handle different formats
                if format_type == Formats.dict:
                    # Convert dict to DataFrame
                    if isinstance(obj, dict):
                        return ibis.memtable(obj)
                    else:
                        raise ValueError(
                            f"Expected dict for dict format, got {type(obj)}"
                        )

                elif format_type == Formats.csv:
                    # Use Ibis read_csv
                    return read_with_format(
                        lambda: ibis.read_csv(obj, **kwargs),
                        "Failed to read CSV with Ibis",
                    )

                elif format_type == Formats.json:
                    # Use Ibis read_json if possible
                    if isinstance(  # TODO(deepyaman): Support `Sequence[str | Path]`
                        obj, (str, Path)
                    ):
                        return read_with_format(
                            lambda: ibis.read_json(obj, **kwargs),
                            "Failed to read JSON with Ibis",
                        )
                    elif isinstance(obj, (list, Mapping)):
                        # If it's a Python object that's JSON-serializable
                        return ibis.memtable(obj)
                    else:
                        raise ValueError(
                            f"Unsupported JSON input type: {type(obj)}"
                        )

                elif format_type == Formats.parquet:
                    # Use Ibis read_parquet
                    return read_with_format(
                        lambda: ibis.read_parquet(obj, **kwargs),
                        "Failed to read Parquet with Ibis",
                    )

                elif format_type in (
                    Formats.feather,
                    Formats.pickle,
                    Formats.json_normalize,
                ):
                    # Formats not natively supported by Ibis
                    raise ValueError(
                        f"{format_type.value} format is not natively supported by Ibis. "
                        "Use a custom callable for from_format instead."
                    )

                else:
                    # For other formats not natively supported by Ibis
                    raise ValueError(  # pragma: no cover
                        f"Format {format_type} is not supported natively by Ibis. "
                        "Use a custom callable for from_format instead."
                    )

        @classmethod
        def to_format(cls, data: ibis.Table, config) -> Any:
            """
            Converts a table to the format specified in the
            :py:class:`pandera.api.ibis.model.DataFrameModel` config options ``to_format``
            and ``to_format_kwargs``.

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
            else:
                # Handle different formats natively using Ibis
                try:
                    format_type = Formats(config.to_format)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported format: {config.to_format}. "
                        f"Ibis natively supports: dict."
                    ) from exc

                # Handle specific formats
                if format_type == Formats.dict:
                    # Convert to dict
                    return data.to_pyarrow().to_pydict()

                elif format_type in (
                    Formats.csv,
                    Formats.json,
                    Formats.feather,
                    Formats.parquet,
                    Formats.pickle,
                    Formats.json_normalize,
                ):
                    # Formats not natively supported by Ibis
                    raise ValueError(
                        f"{format_type.value} format is not natively supported by Ibis. "
                        "Use a custom callable for to_format instead."
                    )

                else:
                    # For other formats not natively supported by Ibis
                    raise ValueError(  # pragma: no cover
                        f"Format {format_type} is not supported natively by Ibis. "
                        "Use a custom callable for to_format instead."
                    )

        @classmethod
        def _get_schema_model(cls, field):
            if not field.sub_fields:
                raise TypeError(
                    "Expected a typed pandera.typing.ibis.Table,"
                    " e.g. Table[Schema]"
                )
            schema_model = field.sub_fields[0].type_
            return schema_model
