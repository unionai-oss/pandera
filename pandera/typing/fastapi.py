"""FastAPI-specific types."""

import functools
from typing import Any, BinaryIO, Generic, Type

from pandera.engines import PYDANTIC_V2
from pandera.typing.common import T

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

try:
    import fastapi
    import starlette.datastructures

    FASTAPI_INSTALLED = True
except ImportError:
    FASTAPI_INSTALLED = False


if PYDANTIC_V2:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema


if FASTAPI_INSTALLED:
    # pylint: disable=too-few-public-methods
    class UploadFile(fastapi.UploadFile, Generic[T]):
        """Pandera-specific subclass of fastapi.UploadFile.

        This type uses :py:class:`pandera.typing.DataFrame` to read files into
        dataframe format based on the
        :py:class:`pandera.api.pandas.models.DataFrameModel` configuration.
        """

        def __init__(
            self,
            data: Any,
            filename: str,
            file: BinaryIO,
            *args,
            **kwargs,
        ):
            """
            Initialize UploadFile object that has a ``data`` property that
            contains validated data.

            :param data: pandera-validated data
            :filename: name of file©
            :file: a file-like object
            """
            self.data = data
            super().__init__(file=file, filename=filename, *args, **kwargs)

        if PYDANTIC_V2:

            # pylint: disable=unused-argument
            @classmethod
            def __get_pydantic_core_schema__(
                cls, _source_type: Any, _handler: GetCoreSchemaHandler
            ) -> core_schema.CoreSchema:
                dataframe_type = get_args(_source_type)[0]
                return core_schema.no_info_plain_validator_function(
                    functools.partial(
                        cls.pydantic_validate_v2,
                        dataframe_type=dataframe_type,
                    )
                )

            @classmethod
            def pydantic_validate_v2(
                cls: Type["UploadFile"], obj: Any, dataframe_type
            ) -> Any:
                """
                Pydantic validation method for validating dataframes in the context
                of a file upload.
                """
                if not isinstance(obj, starlette.datastructures.UploadFile):
                    raise ValueError(
                        f"Expected UploadFile, received: {type(obj)}"
                    )

                schema_model = get_args(dataframe_type)[0]
                validated_data = dataframe_type.pydantic_validate(
                    obj.file, schema_model
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
