"""Declarative TensorDict models."""

import sys
from typing import Any, ClassVar, Union, cast

from pandera.api.base.model import BaseModel
from pandera.api.checks import Check
from pandera.api.dataframe.model_components import CHECK_KEY, Field, FieldInfo
from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.container import TensorDictSchema
from pandera.api.tensordict.model_config import BaseConfig
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo

_CONFIG_KEY = "Config"

try:
    from typing_extensions import Self
except ImportError:
    from typing import Self


def _is_field(name: str) -> bool:
    return not name.startswith("_") and name != _CONFIG_KEY


class TensorDictModel(BaseModel):
    """Model of a TensorDict schema.

    *new in 0.19.0*

    See the :ref:`User Guide <tensordict-models>` for more.
    """

    Config: type[BaseConfig] = BaseConfig

    __fields__: ClassVar[dict[str, tuple[AnnotationInfo, FieldInfo]]] = {}

    @classmethod
    def build_schema_(cls, **kwargs) -> TensorDictSchema:
        columns = cls._build_columns(
            cls.__fields__,
            cls.__checks__,
        )
        return TensorDictSchema(
            keys=columns,
            checks=cls.__root_checks__,
            batch_size=cls.__config__.batch_size,
            name=getattr(cls.__config__, "name", None),
            title=getattr(cls.__config__, "title", None),
            description=getattr(cls.__config__, "description", None),
            coerce=getattr(cls.__config__, "coerce", False),
            **kwargs,
        )

    @classmethod
    def _build_columns(
        cls,
        fields: dict[str, tuple[AnnotationInfo, FieldInfo]],
        checks: dict[str, list[Check]],
    ) -> dict[str, Tensor]:
        columns = {}
        for name, (annotation, field) in fields.items():
            dtype = annotation.raw_annotation
            if dtype is None:
                raise SchemaInitError(
                    f"expected annotation for field '{name}'"
                )

            field_checks = checks.get(name, [])
            column_kwargs = field.to_tensor_kwargs(
                dtype,
                optional=not annotation.optional,
                checks=field_checks,
            )
            column = Tensor(**column_kwargs)
            columns[name] = column

        return columns

    @classmethod
    def _collect_fields(cls) -> dict[str, tuple[AnnotationInfo, FieldInfo]]:
        import inspect

        annotations = inspect.get_annotations(cls)
        attrs = vars(cls)

        missing = []
        for name, attr in attrs.items():
            if inspect.isroutine(attr):
                continue
            if not _is_field(name):
                annotations.pop(name, None)
            elif name not in annotations:
                missing.append(name)

        if missing:
            raise SchemaInitError(f"Found missing annotations: {missing}")

        fields = {}
        for field_name, annotation in annotations.items():
            if not _is_field(field_name):
                continue
            field = attrs.get(field_name)
            if field is None:
                field = TensorDictField()
            elif CHECK_KEY in vars(field):
                field = field[CHECK_KEY]
            fields[field_name] = (AnnotationInfo(annotation), field)
        return fields

    @classmethod
    def _collect_config_and_extras(cls):
        """Collect config options from class."""
        if "Config" in cls.__dict__:
            cls.Config.name = (
                cls.Config.name
                if hasattr(cls.Config, "name")
                else cls.__name__
            )
        else:
            cls.Config = type("Config", (cls.Config,), {"name": cls.__name__})

        config_options = {}
        extras = {}
        for name, value in vars(cls.Config).items():
            if _is_field(name):
                config_options[name] = value
            elif _is_field(name):
                extras[name] = value

        return cls.Config, extras

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        import inspect

        subclass_annotations = inspect.get_annotations(cls)

        for field_name in subclass_annotations.keys():
            if _is_field(field_name) and field_name not in cls.__dict__:
                field = TensorDictField()
                field.__set_name__(cls, field_name)
                setattr(cls, field_name, field)

        cls.__config__, cls.__extras__ = cls._collect_config_and_extras()
        cls.__fields__ = cls._collect_fields()
        cls.__checks__ = {}
        cls.__root_checks__ = []

    @classmethod
    def to_schema(cls, **kwargs) -> TensorDictSchema:
        return cls.build_schema_(**kwargs)

    @classmethod
    def validate(cls, check_obj, *args, **kwargs):
        schema = cls.to_schema()
        return schema.validate(check_obj, *args, **kwargs)

    def __new__(cls, *args, **kwargs) -> Any:
        if cls is TensorDictModel:
            raise NotImplementedError(
                "TensorDictModel cannot be instantiated directly. "
                "Create a subclass with field definitions."
            )
        return super().__new__(cls)


def TensorDictField():
    """Field specification for TensorDict models."""
    from pandera.api.tensordict.model_components import Field

    return Field()
