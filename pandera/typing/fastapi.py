from typing import Any, Callable, Generic, Iterable, Type

import fastapi
import pandas as pd
import starlette.datastructures

from ..errors import SchemaError, SchemaInitError
from .common import T
from .formats import Formats

try:
    from pydantic.fields import ModelField
except ImportError:
    ModelField = Any  # type: ignore


class UploadFile(fastapi.UploadFile, Generic[T]):
    @classmethod
    def __get_validators__(
        cls: Type["UploadFile"],
    ) -> Iterable[Callable[..., Any]]:
        yield cls.validate

    @classmethod
    def validate(cls: Type["UploadFile"], obj: Any, field: ModelField) -> Any:
        if not isinstance(obj, starlette.datastructures.UploadFile):
            raise ValueError(f"Expected UploadFile, received: {type(obj)}")

        df_model_field = field.sub_fields[0]
        df_type = df_model_field.type_
        return df_type._pydantic_validate(obj.file, df_model_field)
