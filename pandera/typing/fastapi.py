"""FastAPI-specific types."""

from typing import Any, Callable, Generic, Iterable, Type

from .common import T

try:
    from pydantic.fields import ModelField
except ImportError:
    ModelField = Any  # type: ignore


try:
    import fastapi
    import starlette.datastructures

    FASTAPI_INSTALLED = True
except ImportError:
    FASTAPI_INSTALLED = False


if FASTAPI_INSTALLED:
    # pylint: disable=too-few-public-methods
    class UploadFile(fastapi.UploadFile, Generic[T]):
        """Pandera-specific subclass of fastapi.UploadFile.

        This type uses :py:class:`pandera.typing.DataFrame` to read files into
        dataframe format based on the :py:class:`pandera.models.SchemaModel`
        configuration.
        """

        __slots__ = ("data",)

        def __init__(self, data: Any, filename: str, file, *args, **kwargs):
            """
            Initialize UploadFile object that has a ``data`` property that
            contains validated data.

            :param data: pandera-validated data
            :filename:
            """
            super().__init__(filename, file, *args, **kwargs)
            self.data = data

        @classmethod
        def __get_validators__(
            cls: Type["UploadFile"],
        ) -> Iterable[Callable[..., Any]]:
            """Pydantic method for yielding validators."""
            yield cls.pydantic_validate

        @classmethod
        def pydantic_validate(
            cls: Type["UploadFile"], obj: Any, field: ModelField
        ) -> Any:
            """
            Pydantic validation method for validating dataframes in the context
            of a file upload.
            """
            if not isinstance(obj, starlette.datastructures.UploadFile):
                raise ValueError(f"Expected UploadFile, received: {type(obj)}")

            schema_model_field = field.sub_fields[0]  # type: ignore[index]
            validated_data = schema_model_field.type_.pydantic_validate(
                obj.file, schema_model_field
            )
            obj.file.seek(0)
            return UploadFile(validated_data, obj.filename, obj.file)
