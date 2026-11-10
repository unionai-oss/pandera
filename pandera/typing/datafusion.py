"""Pandera type annotations for DataFusion."""

import functools
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from packaging import version

from pandera.typing.common import DataFrameBase, DataFrameModel
from pandera.typing.formats import Formats

try:
    import datafusion

    DATAFUSION_INSTALLED = True
except ImportError:
    DATAFUSION_INSTALLED = False


def datafusion_version():
    """Return the DataFusion version."""
    return version.parse(datafusion.__version__)


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if DATAFUSION_INSTALLED:

    class DataFrame(DataFrameBase, datafusion.DataFrame, Generic[T]):
        """Annotation-only generic for ``datafusion.DataFrame``."""

        @classmethod
        def from_format(cls, obj: Any, config) -> "datafusion.DataFrame":
            """Convert serialized data into a ``datafusion.DataFrame``.

            The format is taken from the
            :py:class:`pandera.api.datafusion.model.DataFrameModel` config
            options ``from_format`` and ``from_format_kwargs``. File-based
            formats are read through a fresh
            :class:`datafusion.SessionContext`.

            :param obj: object representing a serialized dataframe.
            :param config: dataframe model configuration object.
            """
            if config.from_format is None:
                if isinstance(obj, datafusion.DataFrame):
                    return obj
                ctx = datafusion.SessionContext()
                try:
                    import pyarrow

                    if isinstance(obj, dict):
                        return ctx.from_pydict(obj)
                    if isinstance(obj, pyarrow.Table):
                        return ctx.from_arrow(obj)
                    return ctx.from_arrow(pyarrow.table(obj))
                except Exception as exc:
                    raise ValueError(
                        f"Expected datafusion.DataFrame, found {type(obj)}"
                    ) from exc

            if callable(config.from_format):
                reader = config.from_format
                return reader(obj, **(config.from_format_kwargs or {}))

            try:
                format_type = Formats(config.from_format)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported format: {config.from_format}. "
                    "DataFusion natively supports: dict, csv, json, parquet, "
                    "and feather."
                ) from exc

            kwargs = config.from_format_kwargs or {}
            ctx = datafusion.SessionContext()

            if format_type == Formats.dict:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected dict for dict format, got {type(obj)}"
                    )
                return ctx.from_pydict(obj)

            try:
                if format_type == Formats.csv:
                    return ctx.read_csv(obj, **kwargs)
                if format_type == Formats.json:
                    return ctx.read_json(obj, **kwargs)
                if format_type == Formats.parquet:
                    return ctx.read_parquet(obj, **kwargs)
                if format_type == Formats.feather:
                    from pyarrow import feather

                    return ctx.from_arrow(feather.read_table(obj, **kwargs))
            except Exception as exc:
                raise ValueError(
                    f"Failed to read {format_type.value} with DataFusion: "
                    f"{exc}"
                ) from exc

            raise ValueError(
                f"{format_type.value} format is not natively supported by "
                "DataFusion. Use a custom callable for from_format instead."
            )

        @classmethod
        def to_format(cls, data: "datafusion.DataFrame", config) -> Any:
            """Convert a dataframe to the format specified in the model config.

            Driven by the
            :py:class:`pandera.api.datafusion.model.DataFrameModel` config
            options ``to_format`` and ``to_format_kwargs``. File-based
            formats execute the plan and write through DataFusion; pass the
            destination in ``to_format_kwargs`` as ``path`` (``dest`` for
            feather).

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
                    "DataFusion natively supports: dict, csv, json, parquet, "
                    "and feather."
                ) from exc

            kwargs = config.to_format_kwargs or {}

            if format_type == Formats.dict:
                return data.to_arrow_table().to_pydict()
            if format_type == Formats.csv:
                # DataFusion omits the header row by default, which
                # ``from_format="csv"`` (and most other readers) expect.
                return data.write_csv(**{"with_header": True, **kwargs})
            if format_type == Formats.json:
                return data.write_json(**kwargs)
            if format_type == Formats.parquet:
                return data.write_parquet(**kwargs)
            if format_type == Formats.feather:
                from pyarrow import feather

                return feather.write_feather(data.to_arrow_table(), **kwargs)

            raise ValueError(
                f"{format_type.value} format is not natively supported by "
                "DataFusion. Use a custom callable for to_format instead."
            )

        @classmethod
        def _get_schema_model(cls, field):
            if not field.sub_fields:
                raise TypeError(
                    "Expected a typed pandera.typing.datafusion.DataFrame,"
                    " e.g. DataFrame[Schema]"
                )
            return field.sub_fields[0].type_
