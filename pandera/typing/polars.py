"""Pandera type annotations for Polars."""

import functools
import io
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from packaging import version

from pandera.config import config_context
from pandera.engines import PYDANTIC_V2
from pandera.errors import SchemaError, SchemaInitError
from pandera.typing.common import DataFrameBase, DataFrameModel, SeriesBase
from pandera.typing.formats import Formats

try:
    import polars as pl

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False


if PYDANTIC_V2:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema


def polars_version():
    """Return the polars version."""
    return version.parse(pl.__version__)


if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if POLARS_INSTALLED:

    class LazyFrame(DataFrameBase, pl.LazyFrame, Generic[T]):
        """
        Pandera generic for pl.LazyFrame, only used for type annotation.

        *new in 0.19.0*
        """

    class DataFrame(DataFrameBase, pl.DataFrame, Generic[T]):
        """
        Pandera generic for pl.DataFrame, only used for type annotation.

        *new in 0.19.0*
        """

        @classmethod
        def from_format(cls, obj: Any, config) -> pl.DataFrame:
            """
            Converts serialized data from a specific format
            specified in the :py:class:`pandera.api.polars.model.DataFrameModel` config options
            ``from_format`` and ``from_format_kwargs``.

            :param obj: object representing a serialized dataframe.
            :param config: dataframe model configuration object.
            """

            if config.from_format is None:
                if not isinstance(obj, pl.DataFrame):
                    try:
                        obj = pl.DataFrame(obj)
                    except Exception as exc:
                        raise ValueError(
                            f"Expected pl.DataFrame, found {type(obj)}"
                        ) from exc
                return obj

            if callable(config.from_format):
                reader = config.from_format
                return reader(obj, **(config.from_format_kwargs or {}))
            else:
                # Handle different formats natively using polars
                try:
                    format_type = Formats(config.from_format)
                except (
                    ValueError
                ) as exc:  # pragma: no cover - tested via mocks
                    raise ValueError(
                        f"Unsupported format: {config.from_format}. "
                        f"Polars natively supports: dict, csv, json, parquet, and feather."
                    ) from exc

                kwargs = config.from_format_kwargs or {}

                # Define a helper function to handle format-specific reading with error handling
                def read_with_format(read_func, error_prefix):
                    """Helper to handle format-specific reading with standardized error handling.

                    Args:
                        read_func: Function to call for reading the data
                        error_prefix: Prefix for error message if the reading fails

                    Returns:
                        DataFrame: The resulting Polars DataFrame

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
                        return pl.DataFrame(obj)
                    else:
                        raise ValueError(
                            f"Expected dict for dict format, got {type(obj)}"
                        )

                elif format_type == Formats.csv:
                    # Use polars read_csv
                    return read_with_format(
                        lambda: pl.read_csv(obj, **kwargs),
                        "Failed to read CSV with polars",
                    )

                elif format_type == Formats.json:
                    # Use polars read_json if possible
                    if isinstance(obj, (str, io.StringIO)):
                        return read_with_format(
                            lambda: pl.read_json(obj, **kwargs),
                            "Failed to read JSON with polars",
                        )
                    elif isinstance(obj, (list, Mapping)):
                        # If it's a Python object that's JSON-serializable
                        return pl.DataFrame(obj)
                    else:
                        raise ValueError(
                            f"Unsupported JSON input type: {type(obj)}"
                        )

                elif format_type == Formats.parquet:
                    # Use polars read_parquet
                    return read_with_format(
                        lambda: pl.read_parquet(obj, **kwargs),
                        "Failed to read Parquet with polars",
                    )

                elif format_type == Formats.feather:
                    # Use polars read_ipc for feather files
                    return read_with_format(
                        lambda: pl.read_ipc(obj, **kwargs),
                        "Failed to read Feather/IPC with polars",
                    )

                elif format_type in (Formats.pickle, Formats.json_normalize):
                    # Formats not natively supported by polars
                    raise ValueError(
                        f"{format_type.value} format is not natively supported by polars. "
                        "Use a custom callable for from_format instead."
                    )

                else:
                    # For other formats not natively supported by polars
                    raise ValueError(  # pragma: no cover
                        f"Format {format_type} is not supported natively by polars. "
                        "Use a custom callable for from_format instead."
                    )

        @classmethod
        def to_format(cls, data: pl.DataFrame, config) -> Any:
            """
            Converts a dataframe to the format specified in the
            :py:class:`pandera.api.polars.model.DataFrameModel` config options ``to_format``
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
                # Handle different formats natively using polars
                try:
                    format_type = Formats(config.to_format)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported format: {config.to_format}. "
                        f"Polars natively supports: dict, csv, json, parquet, and feather."
                    ) from exc

                kwargs = config.to_format_kwargs or {}

                # Helper function for writing to a buffer with error handling
                def write_to_buffer(
                    buffer_factory, write_method, error_prefix
                ):
                    """
                    Helper to write DataFrame to a buffer with standardized error handling.

                    Args:
                        buffer_factory: Function that returns a new buffer
                        write_method: Method to call on the buffer (takes buffer and kwargs)
                        error_prefix: Prefix for error message if writing fails

                    Returns:
                        The buffer or buffer content

                    Raises:
                        ValueError: If writing fails
                    """
                    try:  # pragma: no cover
                        buffer = buffer_factory()
                        write_method(buffer, **kwargs)
                        buffer.seek(0)
                        return (
                            buffer.getvalue()
                            if isinstance(buffer, io.StringIO)
                            else buffer
                        )
                    except Exception as exc:  # pragma: no cover
                        raise ValueError(f"{error_prefix}: {exc}") from exc

                # Handle specific formats
                if format_type == Formats.dict:
                    # Convert to dict
                    return data.to_dict()

                elif format_type == Formats.csv:
                    # Use polars write_csv
                    return write_to_buffer(  # pragma: no cover
                        io.StringIO,
                        data.write_csv,
                        "Failed to write CSV with polars",
                    )

                elif format_type == Formats.json:
                    # Use polars write_json
                    return write_to_buffer(  # pragma: no cover
                        io.StringIO,
                        data.write_json,
                        "Failed to write JSON with polars",
                    )

                elif format_type == Formats.parquet:
                    # Use polars write_parquet
                    return write_to_buffer(  # pragma: no cover
                        io.BytesIO,
                        data.write_parquet,
                        "Failed to write Parquet with polars",
                    )

                elif format_type == Formats.feather:
                    # Use polars write_ipc for feather files
                    return write_to_buffer(  # pragma: no cover
                        io.BytesIO,
                        data.write_ipc,
                        "Failed to write Feather/IPC with polars",
                    )

                elif format_type in (Formats.pickle, Formats.json_normalize):
                    # Formats not natively supported by polars
                    raise ValueError(
                        f"{format_type.value} format is not natively supported by polars. "
                        "Use a custom callable for to_format instead."
                    )

                else:
                    # For other formats not natively supported by polars
                    raise ValueError(  # pragma: no cover
                        f"Format {format_type} is not supported natively by polars. "
                        "Use a custom callable for to_format instead."
                    )

        @classmethod
        def _get_schema_model(cls, field):
            if not field.sub_fields:
                raise TypeError(
                    "Expected a typed pandera.typing.polars.DataFrame,"
                    " e.g. DataFrame[Schema]"
                )
            schema_model = field.sub_fields[0].type_
            return schema_model

        if PYDANTIC_V2:

            @classmethod
            def __get_pydantic_core_schema__(
                cls, _source_type: Any, _handler: GetCoreSchemaHandler
            ) -> core_schema.CoreSchema:
                """
                Generate a Pydantic core schema for Polars DataFrames.

                This method is used by Pydantic v2 to validate and serialize Polars DataFrames.
                It creates a schema that validates input data against the Pandera schema
                and returns a properly validated DataFrame.

                Args:
                    _source_type: The annotated type
                    _handler: Pydantic schema handler

                Returns:
                    CoreSchema: A Pydantic core schema for validation

                Note:
                    Compatible with Pydantic v2.0.0+ and requires Polars 0.19.0+
                """
                # prevent validation in __setattr__ function in DataFrameBase class
                with config_context(validation_enabled=False):
                    schema_model = _source_type().__orig_class__.__args__[0]

                # Extract schema information
                schema = schema_model.to_schema()
                schema_json_columns = schema_model.to_json_schema()[
                    "properties"
                ]

                # Map JSON schema types to Pydantic core schema types
                type_map = {
                    "string": core_schema.str_schema(),
                    "integer": core_schema.int_schema(),
                    "number": core_schema.float_schema(),
                    "boolean": core_schema.bool_schema(),
                    "datetime": core_schema.datetime_schema(),
                }

                # Prepare the validator function
                function = functools.partial(
                    cls.pydantic_validate,
                    schema_model=schema_model,
                )

                # Generate the input schema for use in validation and serialization
                json_schema_input_schema = core_schema.list_schema(
                    core_schema.typed_dict_schema(
                        {
                            key: core_schema.typed_dict_field(
                                type_map[
                                    schema_json_columns[key]["items"]["type"]
                                ]
                            )
                            for key in schema.dtypes.keys()
                        },
                    )
                )

                try:
                    # The json_schema_input_schema parameter is only available in
                    # pydantic_core >=2.30.0. On earlier versions, we'll fall back
                    # to the simpler validation function.
                    return core_schema.no_info_plain_validator_function(
                        function,
                        json_schema_input_schema=json_schema_input_schema,
                        serialization=core_schema.plain_serializer_function_ser_schema(
                            function=lambda df: df,
                            info_arg=False,
                            return_schema=json_schema_input_schema,
                        ),
                    )
                except TypeError:
                    # Fallback for older pydantic_core versions
                    return core_schema.no_info_plain_validator_function(
                        function
                    )

        else:

            @classmethod
            def __get_validators__(cls):  # pragma: no cover
                yield cls._pydantic_validate

        @classmethod
        def pydantic_validate(cls, obj: Any, schema_model) -> pl.DataFrame:
            """
            Verify that the input can be converted into a polars dataframe that
            meets all schema requirements.

            This is for pydantic >= v2
            """
            try:
                schema = schema_model.to_schema()
            except SchemaInitError as exc:
                raise ValueError(
                    f"Cannot use {cls.__name__} as a pydantic type as its "
                    "DataFrameModel cannot be converted to a DataFrameSchema.\n"
                    f"Please revisit the model to address the following errors:"
                    f"\n{exc}"
                ) from exc

            data = cls.from_format(obj, schema_model.__config__)

            try:
                valid_data = schema.validate(data)
            except SchemaError as exc:
                raise ValueError(str(exc)) from exc

            return cls.to_format(valid_data, schema_model.__config__)

        @classmethod
        def _pydantic_validate(cls, obj: Any, field) -> pl.DataFrame:
            """
            Verify that the input can be converted into a polars dataframe that
            meets all schema requirements.

            This is for pydantic < v1
            """
            schema_model = cls._get_schema_model(field)
            return cls.pydantic_validate(obj, schema_model)

    class Series(SeriesBase, pl.Series, Generic[T]):
        """
        Pandera generic for pl.Series, only used for type annotation.

        *new in 0.19.0*
        """
