"""Class-based API for PySpark models."""

import copy
import inspect
import re
import typing
from collections.abc import Callable, Iterable, Mapping
from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
    cast,
)

from pyspark.sql.types import StructType
from typing_extensions import Self

from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model_components import Field, FieldInfo
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.typing.common import DataFrameBase
from pandera.typing.pyspark import DataFrame as PySparkPandasDataFrame
from pandera.typing.pyspark_sql import DataFrame as PySparkSQLDataFrame

from .components import Column
from .container import DataFrameSchema
from .model_config import BaseConfig
from .types import PySparkFrame

DataFrame = Union[PySparkPandasDataFrame, PySparkSQLDataFrame]

_CONFIG_KEY = "Config"

MODEL_CACHE: dict[type["DataFrameModel"], DataFrameSchema] = {}
GENERIC_SCHEMA_CACHE: dict[
    tuple[type["DataFrameModel"], tuple[type[Any], ...]],
    type["DataFrameModel"],
] = {}

F = TypeVar("F", bound=Callable)


def docstring_substitution(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper to substitute the doc strings."""
    if args and kwargs:
        raise AssertionError(
            "Either positional args or keyword args are accepted"
        )
    params = args or kwargs

    def decorator(func: F) -> F:
        func.__doc__ = func.__doc__ and func.__doc__ % params
        return cast(F, func)

    return decorator


def _is_field(name: str) -> bool:
    """Ignore private and reserved keywords."""
    return not name.startswith("_") and name != _CONFIG_KEY


_config_options = [attr for attr in vars(BaseConfig) if _is_field(attr)]


def _extract_config_options_and_extras(
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_options, extras = {}, {}
    for name, value in vars(config).items():
        if name in _config_options:
            config_options[name] = value
        elif _is_field(name):
            extras[name] = value
        # drop private/reserved keywords

    return config_options, extras


def _convert_extras_to_checks(extras: dict[str, Any]) -> list[Check]:
    """
    New in GH#383.
    Any key not in BaseConfig keys is interpreted as defining a dataframe check. This function
    defines this conversion as follows:
        - Look up the key name in Check
        - If value is
            - tuple: interpret as args
            - dict: interpret as kwargs
            - anything else: interpret as the only argument to pass to Check
    """
    checks = []
    for name, value in extras.items():
        if isinstance(value, tuple):
            args, kwargs = value, {}
        elif isinstance(value, dict):
            args, kwargs = (), value
        else:
            args, kwargs = (value,), {}

        # dispatch directly to getattr to raise the correct exception
        checks.append(Check.__getattr__(name)(*args, **kwargs))

    return checks


class DataFrameModel(_DataFrameModel[PySparkFrame, DataFrameSchema]):
    """Definition of a :class:`~pandera.api.pyspark.container.DataFrameSchema`.

    *new in 0.16.0*

    See the :ref:`User Guide <dataframe-models>` for more.
    """

    Config: type[BaseConfig] = BaseConfig
    __extras__: dict[str, Any] | None = None
    __config__: type[BaseConfig] | None = None

    @classmethod
    @docstring_substitution(validate_doc=BaseSchema.validate.__doc__)
    def validate(
        cls: type[Self],
        check_obj: DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = True,
        inplace: bool = False,
    ) -> DataFrameBase[Self]:
        """%(validate_doc)s"""
        return cast(
            DataFrameBase[Self],
            cls.to_schema().validate(
                check_obj,
                head,
                tail,
                sample,
                random_state,
                lazy,
                inplace,  # type: ignore
            ),
        )

    @docstring_substitution(validate_doc=DataFrameSchema.validate.__doc__)
    def __new__(cls, *args, **kwargs) -> DataFrameBase[Self]:  # type: ignore [misc]
        """%(validate_doc)s"""
        return cast(DataFrameBase[Self], cls.validate(*args, **kwargs))

    def __init_subclass__(cls, **kwargs):
        """Ensure :class:`~pandera.api.pyspark.model_components.FieldInfo` instances."""
        if "Config" in cls.__dict__:
            cls.Config.name = (
                cls.Config.name
                if hasattr(cls.Config, "name")
                else cls.__name__
            )
        else:
            cls.Config = type("Config", (BaseConfig,), {"name": cls.__name__})
        super().__init_subclass__(**kwargs)
        subclass_annotations = cls.__dict__.get("__annotations__", {})
        for field_name in subclass_annotations.keys():
            if _is_field(field_name) and field_name not in cls.__dict__:
                # Field omitted
                field = Field()
                field.__set_name__(cls, field_name)
                setattr(cls, field_name, field)

        cls.__config__, cls.__extras__ = cls._collect_config_and_extras()

    def __class_getitem__(
        cls: type[Self],
        params: Union[type[Any], tuple[type[Any], ...]],
    ) -> type[Self]:
        """Parameterize the class's generic arguments with the specified types"""
        if not hasattr(cls, "__parameters__"):
            raise TypeError(
                f"{cls.__name__} must inherit from typing.Generic before being parameterized"
            )

        __parameters__: tuple[TypeVar, ...] = cls.__parameters__  # type: ignore

        if not isinstance(params, tuple):
            params = (params,)
        if len(params) != len(__parameters__):
            raise ValueError(
                f"Expected {len(__parameters__)} generic arguments but found {len(params)}"
            )
        if (cls, params) in GENERIC_SCHEMA_CACHE:
            return typing.cast(type[Self], GENERIC_SCHEMA_CACHE[(cls, params)])

        param_dict: dict[TypeVar, type[Any]] = dict(
            zip(__parameters__, params)
        )
        extra: dict[str, Any] = {"__annotations__": {}}
        for field, (annot_info, field_info) in cls._collect_fields().items():
            if isinstance(annot_info.arg, TypeVar):
                if annot_info.arg in param_dict:
                    raw_annot = annot_info.origin[param_dict[annot_info.arg]]  # type: ignore
                    if annot_info.optional:
                        raw_annot = Optional[raw_annot]  # noqa: UP045
                    extra["__annotations__"][field] = raw_annot
                    extra[field] = copy.deepcopy(field_info)

        parameterized_name = (
            f"{cls.__name__}[{', '.join(p.__name__ for p in params)}]"
        )
        parameterized_cls = type(parameterized_name, (cls,), extra)
        GENERIC_SCHEMA_CACHE[(cls, params)] = parameterized_cls
        return parameterized_cls

    @classmethod
    def build_schema_(cls, **kwargs):
        return DataFrameSchema(
            cls._build_columns(cls.__fields__, cls.__checks__),
            checks=cls.__root_checks__,  # type: ignore
            **kwargs,  # type: ignore
        )

    @classmethod
    def to_structtype(cls) -> StructType:
        """Recover fields of DataFrameModel as a PySpark StructType object.

        :returns: StructType object with current model fields.
        """
        return cls.to_schema().to_structtype()

    @classmethod
    def to_ddl(cls) -> str:
        """Recover fields of DataFrameModel as a PySpark DDL string.

        :returns: String with current model fields, in compact DDL format.
        """
        return cls.to_schema().to_ddl()

    @classmethod
    def _build_columns(  # pylint:disable=too-many-locals
        cls,
        fields: dict[str, tuple[AnnotationInfo, FieldInfo]],
        checks: dict[str, list[Check]],
    ) -> dict[str, Column]:
        columns: dict[str, Column] = {}
        for field_name, (annotation, field) in fields.items():
            field_checks = checks.get(field_name, [])
            field_name = field.name
            check_name = getattr(field, "check_name", None)

            if annotation.metadata:
                if field.dtype_kwargs:
                    raise TypeError(
                        "Cannot specify redundant 'dtype_kwargs' "
                        + f"for {annotation.raw_annotation}."
                        + "\n Usage Tip: Drop 'typing.Annotated'."
                    )
                dtype_kwargs = _get_dtype_kwargs(annotation)
                dtype = annotation.arg(**dtype_kwargs)  # type: ignore
            elif annotation.default_dtype:
                dtype = annotation.default_dtype
            else:
                dtype = annotation.arg

            dtype = None if dtype is Any else dtype

            if annotation.origin is None:
                column_kwargs = (
                    field.column_properties(
                        dtype,
                        required=not annotation.optional,
                        checks=field_checks,
                        name=field_name,
                    )
                    if field
                    else {}
                )
                if check_name is False:
                    raise SchemaInitError(
                        f"'check_name' is not supported for {field_name}."
                    )
                # remove unsupported kwargs
                for key in ["parsers", "unique", "default"]:
                    column_kwargs.pop(key, None)
                columns[field_name] = Column(**column_kwargs)
            else:
                raise SchemaInitError(
                    f"Invalid annotation '{field_name}: "
                    f"{annotation.raw_annotation}'"
                )

        return columns


def _get_dtype_kwargs(annotation: AnnotationInfo) -> dict[str, Any]:
    sig = inspect.signature(annotation.arg)  # type: ignore
    dtype_arg_names = list(sig.parameters.keys())
    if len(annotation.metadata) != len(dtype_arg_names):  # type: ignore
        raise TypeError(
            f"Annotation '{annotation.arg.__name__}' requires "  # type: ignore
            + f"all positional arguments {dtype_arg_names}."
        )
    return dict(zip(dtype_arg_names, annotation.metadata))  # type: ignore
