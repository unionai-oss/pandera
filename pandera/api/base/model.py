"""Base classes for model API."""

import os
from collections.abc import Mapping
from typing import (
    Any,
    ClassVar,
    Optional,
    TypeVar,
    Union,
)

from pandera.api.base.model_components import BaseFieldInfo
from pandera.api.base.model_config import BaseModelConfig
from pandera.api.checks import Check
from pandera.typing import AnnotationInfo

TBaseModel = TypeVar("TBaseModel", bound="BaseModel")


class MetaModel(type):
    """Add string representations, mainly for pydantic."""

    def __repr__(cls):
        return str(cls)

    def __str__(cls):
        return cls.__name__


class BaseModel(metaclass=MetaModel):
    """Base class for a Data Object Model."""

    Config: type[BaseModelConfig] = BaseModelConfig
    __extras__: dict[str, Any] | None = None
    __schema__: ClassVar[Any | None] = None
    __config__: type[BaseModelConfig] | None = None

    #: Key according to `FieldInfo.name`
    __fields__: ClassVar[
        Mapping[str, tuple[AnnotationInfo, BaseFieldInfo]]
    ] = {}
    __checks__: ClassVar[dict[str, list[Check]]] = {}
    __root_checks__: ClassVar[list[Check]] = []

    # This is syntantic sugar that delegates to the validate method
    def __new__(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __class_getitem__(
        cls: type[TBaseModel],
        params: Union[type[Any], tuple[type[Any], ...]],
    ) -> type[TBaseModel]:
        """
        Parameterize the class's generic arguments with the specified types.

        This allows for using generic types in the Model class definition.
        """
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> Any:
        """Create a Schema object from this Model class."""
        raise NotImplementedError

    @classmethod
    def to_yaml(cls, stream: os.PathLike | None = None):
        """Convert `Schema` to yaml using `io.to_yaml`."""
        raise NotImplementedError

    @classmethod
    def validate(
        cls: type[TBaseModel],
        check_obj: Any,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Any:
        """Validate a data object."""
        raise NotImplementedError

    @classmethod
    def strategy(cls: type[TBaseModel], *, size: int | None = None):
        """Create a data synthesis strategy."""
        raise NotImplementedError

    @classmethod
    def example(cls: type[TBaseModel], *, size: int | None = None) -> Any:
        """Generate an example of this data model specification."""
        raise NotImplementedError

    ####################################
    # Methods for pydantic integration #
    ####################################

    @classmethod
    def __get_validators__(cls):
        """Yield a pydantic-compatible validator."""
        raise NotImplementedError

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Update pydantic field schema."""
        raise NotImplementedError
