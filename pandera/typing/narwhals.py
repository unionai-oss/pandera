"""Pandera type annotations for Narwhals."""

import functools
import io
from typing import TYPE_CHECKING, Any, Generic, TypeVar, List, Mapping

from pandera.config import config_context
from pandera.engines import PYDANTIC_V2
from pandera.errors import SchemaError, SchemaInitError
from pandera.typing.common import DataFrameBase, DataFrameModel, SeriesBase
from pandera.typing.formats import Formats

try:
    import narwhals as nw

    NARWHALS_INSTALLED = True
except ImportError:
    NARWHALS_INSTALLED = False


if PYDANTIC_V2:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema


def narwhals_version():
    """Return the narwhals version."""
    return nw.__version__


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


if NARWHALS_INSTALLED:
    # pylint: disable=too-few-public-methods
    class LazyFrame(DataFrameBase, nw.LazyFrame, Generic[T]):
        """
        Pandera generic for nw.LazyFrame, only used for type annotation.

        *new in narwhals integration*
        """

    class DataFrame(DataFrameBase, nw.DataFrame, Generic[T]):
        """
        Pandera generic for nw.DataFrame, only used for type annotation.

        *new in narwhals integration*
        """

        @classmethod
        def from_format(cls, obj: Any, config) -> nw.DataFrame[Any]:
            """
            Converts serialized data from a specific format
            specified in the :py:class:`pandera.api.narwhals.model.DataFrameModel` config options
            ``from_format`` and ``from_format_kwargs``.

            :param obj: object representing a serialized dataframe.
            :param config: dataframe model configuration object.
            """
            # Placeholder implementation - would need proper format handling
            if config.from_format is None:
                if hasattr(obj, "__dataframe__"):
                    return nw.from_native(obj, eager_only=True)
                else:
                    # Try to create from dict/list
                    return nw.from_native(obj, eager_only=True)

            if callable(config.from_format):
                reader = config.from_format
                return reader(obj, **(config.from_format_kwargs or {}))
            else:
                # Handle different formats - placeholder implementation
                try:
                    format_type = Formats(config.from_format)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported format: {config.from_format}. "
                        f"Narwhals supports various formats through native backends."
                    ) from exc

                kwargs = config.from_format_kwargs or {}

                # For now, just return the object as-is
                # A full implementation would handle various formats
                return nw.from_native(obj, eager_only=True)

        @classmethod
        def to_format(cls, data: nw.DataFrame[Any], config) -> Any:
            """
            Converts a dataframe to the format specified in the
            :py:class:`pandera.api.narwhals.model.DataFrameModel` config options ``to_format``
            and ``to_format_kwargs``.

            :param data: convert this data to the specified format
            :param config: config object from the DataFrameModel
            """
            # Placeholder implementation
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
                # Handle different formats - placeholder implementation
                try:
                    format_type = Formats(config.to_format)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported format: {config.to_format}. "
                        f"Narwhals supports various formats through native backends."
                    ) from exc

                kwargs = config.to_format_kwargs or {}

                # For now, just return the data as-is
                # A full implementation would handle various formats
                return data

        @classmethod
        def _get_schema_model(cls, field):
            if not field.sub_fields:
                raise TypeError(
                    "Expected a typed pandera.typing.narwhals.DataFrame,"
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
                Generate a Pydantic core schema for Narwhals DataFrames.

                This method is used by Pydantic v2 to validate and serialize Narwhals DataFrames.
                It creates a schema that validates input data against the Pandera schema
                and returns a properly validated DataFrame.

                Args:
                    _source_type: The annotated type
                    _handler: Pydantic schema handler

                Returns:
                    CoreSchema: A Pydantic core schema for validation

                Note:
                    Compatible with Pydantic v2.0.0+ and requires Narwhals integration
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
                                type_map.get(
                                    schema_json_columns[key]["items"]["type"],
                                    core_schema.str_schema(),
                                )
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
        def pydantic_validate(
            cls, obj: Any, schema_model
        ) -> nw.DataFrame[Any]:
            """
            Verify that the input can be converted into a narwhals dataframe that
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
        def _pydantic_validate(cls, obj: Any, field) -> nw.DataFrame[Any]:
            """
            Verify that the input can be converted into a narwhals dataframe that
            meets all schema requirements.

            This is for pydantic < v1
            """
            schema_model = cls._get_schema_model(field)
            return cls.pydantic_validate(obj, schema_model)

    # pylint: disable=too-few-public-methods
    class Series(SeriesBase, nw.Series, Generic[T]):
        """
        Pandera generic for nw.Series, only used for type annotation.

        *new in narwhals integration*
        """
