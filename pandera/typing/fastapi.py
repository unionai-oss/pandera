from typing import Any, Callable, Generic, Iterable, Type

import fastapi
import starlette.datastructures

from .common import T

try:
    from pydantic.fields import ModelField
except ImportError:
    ModelField = Any  # type: ignore


class UploadFile(fastapi.UploadFile, Generic[T]):
    def __init__(self, data, filename, file, *args, **kwargs):
        super().__init__(filename, file, *args, **kwargs)
        self.data = data

    @classmethod
    def __get_validators__(
        cls: Type["UploadFile"],
    ) -> Iterable[Callable[..., Any]]:
        yield cls.pydantic_validate

    @classmethod
    def pydantic_validate(
        cls: Type["UploadFile"], obj: Any, field: ModelField
    ) -> Any:
        if not isinstance(obj, starlette.datastructures.UploadFile):
            raise ValueError(f"Expected UploadFile, received: {type(obj)}")

        schema_model_field = field.sub_fields[0]
        validated_data = schema_model_field.type_.pydantic_validate(
            obj.file, schema_model_field
        )
        obj.file.seek(0)
        return UploadFile(validated_data, obj.filename, obj.file)
