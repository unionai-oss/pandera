"""TensorDictModel for class-based schema definitions."""

from __future__ import annotations

import inspect
import threading
import typing
from typing import Any, ClassVar, cast, get_type_hints

import typing_inspect

from pandera.api.base.model import BaseModel
from pandera.api.checks import Check
from pandera.api.parsers import Parser
from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.container import TensorDictSchema
from pandera.api.tensordict.model_config import BaseConfig
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo

TFields = dict[str, tuple[AnnotationInfo, Any]]
TChecks = dict[str, list[Check]]
TParsers = dict[str, list[Parser]]

_CONFIG_KEY = "Config"
MODEL_CACHE: dict[tuple[type["TensorDictModel"], int], Any] = {}


def _is_field(name: str) -> bool:
    return not name.startswith("_") and name != _CONFIG_KEY


class _FieldsDescriptor:
    def __init__(self) -> None:
        self.cache: dict[type, Any] = {}

    def __get__(self, obj: Any, cls: type[TensorDictModel]) -> TFields:
        if self.cache.get(cls) is None:
            self.cache[cls] = cls._collect_fields()
        return self.cache[cls]


class _SchemaDescriptor:
    def __get__(self, obj: Any, cls: type[TensorDictModel]) -> Any:
        tid = threading.get_ident()
        key = (cls, tid)
        if MODEL_CACHE.get(key) is None:
            try:
                MODEL_CACHE[key] = cls.build_schema_()
            except Exception as e:
                raise AttributeError(
                    f"'{cls.__name__}' does not implement build_schema_() and cannot "
                    f"generate a schema. To be able to generate a schema, subclass the"
                    "TensorDictModel."
                ) from e
        return MODEL_CACHE[key]


class _ChecksDescriptor:
    def __init__(self) -> None:
        self.cache: dict[type, Any] = {}

    def __get__(self, obj: Any, cls: type[TensorDictModel]) -> TChecks:
        if self.cache.get(cls) is None:
            check_infos = typing.cast(
                list[Any], cls._collect_check_infos(CHECK_KEY)
            )
            self.cache[cls] = cls._extract_checks(
                check_infos, field_names=list(cls.__fields__.keys())
            )
        return self.cache[cls]


class _RootCheckDescriptor:
    def __init__(self) -> None:
        self.cache: dict[type, Any] = {}

    def __get__(self, obj: Any, cls: type[TensorDictModel]) -> list[Check]:
        if self.cache.get(cls) is None:
            df_check_infos = cls._collect_check_infos(DATAFRAME_CHECK_KEY)
            custom = cls._extract_df_checks(df_check_infos)
            reg = _convert_extras_to_checks(
                {} if cls.__extras__ is None else cls.__extras__
            )
            self.cache[cls] = custom + reg
        return self.cache[cls]


class _ParsersDescriptor:
    def __init__(self) -> None:
        self.cache: dict[type, Any] = {}

    def __get__(self, obj: Any, cls: type[TensorDictModel]) -> TParsers:
        if self.cache.get(cls) is None:
            parser_infos = typing.cast(
                list[Any], cls._collect_parser_infos(PARSER_KEY)
            )
            self.cache[cls] = cls._extract_parsers(
                parser_infos, field_names=list(cls.__fields__.keys())
            )
        return self.cache[cls]


class _RootParsersDescriptor:
    def __init__(self) -> None:
        self.cache: dict[type, Any] = {}

    def __get__(self, obj: Any, cls: type[TensorDictModel]) -> list[Parser]:
        if self.cache.get(cls) is None:
            df_parser_infos = cls._collect_parser_infos(DATAFRAME_PARSER_KEY)
            self.cache[cls] = cls._extract_df_parsers(df_parser_infos)
        return self.cache[cls]


DATAFRAME_CHECK_KEY = "__dataframe_check_config__"
DATAFRAME_PARSER_KEY = "__dataframe_parser_config__"


def _convert_extras_to_checks(extras: dict[str, Any]) -> list[Check]:
    """Convert extras to checks."""
    from pandera.api.checks import Check

    checks = []
    for name, value in extras.items():
        if isinstance(value, tuple):
            args, kwargs = value, {}
        elif isinstance(value, dict):
            args, kwargs = (), value
        elif value is Ellipsis:
            args, kwargs = (), {}
        else:
            args, kwargs = (value,), {}

        checks.append(getattr(Check, name)(*args, **kwargs))

    return checks


class TensorDictModel(BaseModel):
    """Declarative TensorDict schema using class definitions.

    Fields are defined using ``pa.Field`` directly in type annotations. Use PyTorch dtypes
(e.g., `torch.float32`, `torch.int64`) to specify the expected data type:

    Example:
        >>> import torch
        >>> import pandera.tensordict as pa

        >>> class MySchema(pa.TensorDictModel):
        ...     observation: torch.float32 = pa.Field(shape=(None, 10))
        ...     action: torch.int64 = pa.Field(shape=(None,))
        ...
        ...     class Config:
        ...         batch_size = (32,)

        >>> schema = MySchema.to_schema()
    """

    Config: type[BaseConfig] = BaseConfig
    __schema__ = _SchemaDescriptor()
    __config__: type[BaseConfig] | None = None

    #: Key according to `FieldInfo.name`
    __fields__: ClassVar[TFields] = cast(TFields, _FieldsDescriptor())
    __checks__ = cast(TChecks, _ChecksDescriptor())
    __parsers__: TParsers = cast(TParsers, _ParsersDescriptor())
    __root_checks__ = cast(list[Check], _RootCheckDescriptor())
    __root_parsers__ = cast(list[Parser], _RootParsersDescriptor())

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize subclass and build schema from class annotations."""
        if _CONFIG_KEY in cls.__dict__:
            if not hasattr(cls.Config, "name"):
                cls.Config.name = None
        else:
            cls.Config = type("Config", (cls.Config,), {})
        
        super().__init_subclass__(**kwargs)
        hints = get_type_hints(cls, include_extras=True)
        
        for fname in hints.keys():
            if not _is_field(fname):
                continue
            if fname in cls.__dict__:
                continue
            # Auto-add Field() if not explicitly defined
            from pandera.api.tensordict.model_components import Field
            
            field = Field()
            field.__set_name__(cls, fname)
            setattr(cls, fname, field)
        
        cls.__config__, cls.__extras__ = cls._collect_config_and_extras()

    @classmethod
    def _get_model_attrs(cls) -> dict[str, Any]:
        """Return all attributes from the model class."""
        bases = inspect.getmro(cls)[:-1]
        attrs: dict[str, Any] = {}
        for base in reversed(bases):
            if (
                issubclass(base, TensorDictModel)
                and base is not TensorDictModel
            ):
                attrs.update(base.__dict__)
        return attrs

    @classmethod
    def _collect_fields(cls) -> TFields:
        """Centralize publicly named fields and their corresponding annotations."""
        from pandera.api.tensordict.model_components import TensorDictFieldInfo
        
        annotations = get_type_hints(cls, include_extras=True)
        attrs = cls._get_model_attrs()

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
            field = attrs[field_name]
            
            # Check if it's a TensorDictFieldInfo instance
            if not isinstance(field, TensorDictFieldInfo):
                raise SchemaInitError(
                    f"'{field_name}' can only be assigned a 'Field', "
                    f"not a '{type(field)}'."
                )
            fields[field.name] = (AnnotationInfo(annotation), field)
        return fields

    @classmethod
    def _extract_config_options_and_extras(
        cls,
        config: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract config options and extras from Config class."""
        from pandera.api.dataframe.model_config import BaseConfig as DataFrameBaseConfig
        
        config_options = {}
        extras = {}
        
        # Get allowed config option names
        if hasattr(DataFrameBaseConfig, "__dataclass_fields__"):
            config_field_names = list(
                DataFrameBaseConfig.__dataclass_fields__.keys()
            )
        else:
            config_field_names = [
                attr
                for attr in vars(cls.Config)
                if not attr.startswith("_")
                and callable(getattr(cls.Config, attr)) is False
            ]

        for name, value in vars(config).items():
            if name in config_field_names:
                config_options[name] = value
            elif _is_field(name):
                extras[name] = value

        return config_options, extras

    @classmethod
    def _collect_config_and_extras(
        cls,
    ) -> tuple[type[BaseConfig], dict[str, Any]]:
        """Collect config options from bases, splitting off unknown options."""
        bases = inspect.getmro(cls)[:-1]
        bases = tuple(
            base for base in bases if issubclass(base, TensorDictModel)
        )
        root_model, *models = reversed(bases)

        options, extras = cls._extract_config_options_and_extras(
            root_model.Config
        )

        for model in models:
            config = getattr(model, _CONFIG_KEY, {})
            base_options, base_extras = cls._extract_config_options_and_extras(
                config
            )
            options.update(base_options)
            extras.update(base_extras)

        return type("Config", (cls.Config,), options), extras

    @classmethod
    def _collect_check_infos(cls, key: str) -> list[Any]:
        """Collect inherited check metadata from bases."""
        bases = inspect.getmro(cls)[:-2]
        bases = tuple(
            base for base in bases if issubclass(base, TensorDictModel)
        )

        method_names = set()
        check_infos = []
        for base in bases:
            for attr_name, attr_value in vars(base).items():
                check_info = getattr(attr_value, key, None)
                if not isinstance(check_info, type(cls).__dict__.get("Field", object).__bases__[0] if hasattr(type(cls).__dict__.get("Field", object), "__bases__") else object):
                    continue
                if attr_name in method_names:
                    continue
                method_names.add(attr_name)
                check_infos.append(check_info)
        return check_infos

    @classmethod
    def _collect_parser_infos(cls, key: str) -> list[Any]:
        """Collect inherited parser metadata from bases."""
        bases = inspect.getmro(cls)[:-2]
        bases = tuple(
            base for base in bases if issubclass(base, TensorDictModel)
        )

        method_names = set()
        parser_infos = []
        for base in bases:
            for attr_name, attr_value in vars(base).items():
                parser_info = getattr(attr_value, key, None)
                if not isinstance(parser_info, type(cls).__dict__.get("Field", object).__bases__[0] if hasattr(type(cls).__dict__.get("Field", object), "__bases__") else object):
                    continue
                method_names.add(attr_name)
                parser_infos.append(parser_info)
        return parser_infos

    @staticmethod
    def _regex_filter(seq: Any, regexps: Any) -> set[str]:
        """Filter items matching at least one of the regexes."""
        import re

        matched: set[str] = set()
        for pattern_str in regexps:
            pattern = re.compile(pattern_str)
            matched.update(filter(pattern.match, seq))
        return matched

    @classmethod
    def _extract_checks(
        cls, check_infos: list[Any], field_names: list[str]
    ) -> dict[str, list[Check]]:
        """Collect field annotations from bases in mro reverse order."""
        checks: dict[str, list[Check]] = {}
        for check_info in check_infos:
            if not hasattr(check_info, "fields"):
                continue
                
            check_info_fields = {
                field.name
                if isinstance(field, type(cls).__dict__.get("Field", object).__bases__[0] if hasattr(type(cls).__dict__.get("Field", object), "__bases__") else object)
                else field
                for field in check_info.fields
            }

            matched = (
                cls._regex_filter(field_names, check_info_fields)
                if hasattr(check_info, "regex") and check_info.regex
                else check_info_fields
            )

            check_ = (
                check_info.to_check(cls) if hasattr(check_info, "to_check") else None
            )

            for field in matched:
                if field not in field_names:
                    raise SchemaInitError(
                        f"Check {check_.name if check_ else 'unknown'} is assigned to a non-existing field '{field}'."
                    )
                if field not in checks:
                    checks[field] = []
                checks[field].append(check_)
        return checks

    @classmethod
    def _extract_df_checks(cls, check_infos: list[Any]) -> list[Check]:
        """Collect dataframe-level checks."""
        return [
            check_info.to_check(cls)
            for check_info in check_infos
            if hasattr(check_info, "to_check")
        ]

    @classmethod
    def _extract_parsers(
        cls, parser_infos: list[Any], field_names: list[str]
    ) -> dict[str, list[Parser]]:
        """Collect field parsers from bases in mro reverse order."""
        parsers: dict[str, list[Parser]] = {}
        for parser_info in parser_infos:
            if not hasattr(parser_info, "fields"):
                continue
                
            parser_info_fields = {
                field.name
                if isinstance(field, type(cls).__dict__.get("Field", object).__bases__[0] if hasattr(type(cls).__dict__.get("Field", object), "__bases__") else object)
                else field
                for field in parser_info.fields
            }

            matched = (
                cls._regex_filter(field_names, parser_info_fields)
                if hasattr(parser_info, "regex") and parser_info.regex
                else parser_info_fields
            )

            parser_ = (
                parser_info.to_parser(cls)
                if hasattr(parser_info, "to_parser")
                else None
            )

            for field in matched:
                if field not in field_names:
                    raise SchemaInitError(
                        f"Parser {parser_.name if parser_ else 'unknown'} is assigned to a non-existing field '{field}'."
                    )
                if field not in parsers:
                    parsers[field] = []
                parsers[field].append(parser_)
        return parsers

    @classmethod
    def _extract_df_parsers(cls, parser_infos: list[Any]) -> list[Parser]:
        """Collect dataframe-level parsers."""
        return [
            parser_info.to_parser(cls)
            for parser_info in parser_infos
            if hasattr(parser_info, "to_parser")
        ]

    @classmethod
    def build_schema_(cls, **kwargs) -> TensorDictSchema:
        """Create TensorDictSchema from model class.

        :returns: TensorDictSchema with fields converted to Tensor components.
        """
        cfg = cls.__config__
        fields = cls.__fields__

        columns: dict[str, Any] = {}

        for fname, (ann, fi) in fields.items():
            # Get dtype from annotation
            dtype = ann.arg

            # Handle DataType special case - convert to torch.dtype if possible
            if dtype is not None:
                try:
                    from pandera.engines.tensordict_engine import (
                        DataType as _DataType,
                    )

                    if isinstance(dtype, type) and issubclass(dtype, _DataType):
                        dtype = dtype.type
                except Exception:
                    pass

            # Get shape from Field info
            shape = fi.shape if hasattr(fi, "shape") else None

            # Build tensor kwargs
            tensor_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "shape": shape,
                "name": fname,
            }

            if hasattr(fi, "checks") and fi.checks:
                tensor_kwargs["checks"] = fi.checks

            columns[fname] = Tensor(**tensor_kwargs)

        # Get batch_size from Config
        batch_size: tuple[int | None, ...] | None = (
            getattr(cfg, "batch_size", None) if cfg else None
        )

        try:
            return TensorDictSchema(keys=columns or None, batch_size=batch_size)
        except ImportError as e:
            raise RuntimeError(
                "Could not import TensorDictSchema"
            ) from e

    @classmethod
    def to_schema(cls: type[TensorDictModel]) -> TensorDictSchema:
        """Create :class:`~pandera.TensorDictSchema` from the :class:`.TensorDictModel`."""
        return cls.__schema__

    @classmethod
    def validate(
        cls: type[TensorDictModel],
        check_obj: Any,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Any:
        """Validate data against model schema.

        :param check_obj: TensorDict or tensorclass to validate.
        :returns: Validated data object.
        """
        if cls.__schema__ is not None:
            return cls.__schema__.validate(
                check_obj=check_obj,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
        raise RuntimeError("Model schema was not initialized")

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Validate data against model schema."""
        return cls.validate(*args, **kwargs)
