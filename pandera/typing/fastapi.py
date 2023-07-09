"""FastAPI-specific types."""

from typing import Any, BinaryIO, Callable, Generic, Iterable, Type

from pandera.engines import PYDANTIC_V2
from pandera.typing.common import T

try:
    import fastapi
    import starlette.datastructures

    FASTAPI_INSTALLED = True
except ImportError:
    FASTAPI_INSTALLED = False


if PYDANTIC_V2:
    from pydantic_core import core_schema
    from pydantic import GetJsonSchemaHandler, GetCoreSchemaHandler


if FASTAPI_INSTALLED:
    # pylint: disable=too-few-public-methods
    class UploadFile(fastapi.UploadFile, Generic[T]):
        """Pandera-specific subclass of fastapi.UploadFile.

        This type uses :py:class:`pandera.typing.DataFrame` to read files into
        dataframe format based on the
        :py:class:`pandera.api.pandas.models.DataFrameModel` configuration.
        """

        __slots__ = (
            "data",
            "filename",
            "file",
            "headers",
        )

        def __init__(
            self,
            pandera_data: Any,
            filename: str,
            file: BinaryIO,
            *args,
            **kwargs,
        ):
            """
            Initialize UploadFile object that has a ``data`` property that
            contains validated data.

            :param data: pandera-validated data
            :filename: name of file
            :file: a file-like object
            """
            self.pandera_data = pandera_data
            super().__init__(file=file, filename=filename, *args, **kwargs)

        if PYDANTIC_V2:
            from pydantic import FieldValidationInfo

            @classmethod
            def __get_pydantic_core_schema__(
                cls, _source_type: Any, _handler: GetCoreSchemaHandler
            ) -> core_schema.CoreSchema:
                return core_schema.general_plain_validator_function(
                    cls.pydantic_validate_v2,
                )

            @classmethod
            def pydantic_validate_v2(
                cls: Type["UploadFile"], obj: Any, info: FieldValidationInfo
            ) -> Any:
                """
                Pydantic validation method for validating dataframes in the context
                of a file upload.
                """
                if not isinstance(obj, starlette.datastructures.UploadFile):
                    raise ValueError(
                        f"Expected UploadFile, received: {type(obj)}"
                    )

                schema_model_field = field.sub_fields[0]  # type: ignore[index]
                cls.model_fields
                validated_data = schema_model_field.type_._pydantic_validate(
                    obj.file, schema_model_field
                )
                obj.file.seek(0)
                return UploadFile(validated_data, obj.filename, obj.file)

        else:
            from pydantic.fields import ModelField

            @classmethod
            def __get_validators__(cls):
                yield cls.pydantic_validate_v1

            @classmethod
            def pydantic_validate_v1(
                cls: Type["UploadFile"], obj: Any, field: ModelField
            ) -> Any:
                """
                Pydantic validation method for validating dataframes in the context
                of a file upload.
                """
                if not isinstance(obj, starlette.datastructures.UploadFile):
                    raise ValueError(
                        f"Expected UploadFile, received: {type(obj)}"
                    )

                schema_model_field = field.sub_fields[0]  # type: ignore[index]
                validated_data = schema_model_field.type_._pydantic_validate(
                    obj.file, schema_model_field
                )
                obj.file.seek(0)
                return UploadFile(validated_data, obj.filename, obj.file)
