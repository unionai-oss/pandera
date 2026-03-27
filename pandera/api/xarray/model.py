"""Declarative :class:`DataArrayModel` and :class:`DatasetModel`."""

from __future__ import annotations

import inspect
import threading
import typing
from typing import Any, ClassVar, cast, get_args, get_origin

import typing_inspect

from pandera.api.base.model import BaseModel
from pandera.api.checks import Check
from pandera.api.dataframe.model import _convert_extras_to_checks
from pandera.api.dataframe.model_components import (
    CHECK_KEY,
    DATAFRAME_CHECK_KEY,
    DATAFRAME_PARSER_KEY,
    PARSER_KEY,
    CheckInfo,
    FieldCheckInfo,
    FieldInfo,
    FieldParserInfo,
    ParserInfo,
)
from pandera.api.parsers import Parser
from pandera.api.xarray.components import Coordinate, DataVar
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.api.xarray.model_components import Field as XarrayField
from pandera.api.xarray.model_components import XarrayFieldInfo
from pandera.api.xarray.model_config import DataArrayConfig, DatasetConfig
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.utils import docstring_substitution

_CONFIG_KEY = "Config"


def _is_field(name: str) -> bool:
    return not name.startswith("_") and name != _CONFIG_KEY


def _unwrap_optional(raw: Any) -> Any:
    if typing_inspect.is_optional_type(raw):
        return get_args(raw)[0]
    return raw


def _is_coordinate_annotation(raw: Any) -> bool:
    from pandera.typing.xarray import Coordinate as CoordMark

    raw = _unwrap_optional(raw)
    origin = get_origin(raw)
    return origin is CoordMark


def _coordinate_dtype_arg(raw: Any) -> Any:
    raw = _unwrap_optional(raw)
    args = get_args(raw)
    return args[0] if args else Any


def _all_config_keys(config_cls: type) -> frozenset[str]:
    keys: set[str] = set()
    for base in inspect.getmro(config_cls):
        if base is object:
            continue
        keys.update(k for k in base.__dict__ if _is_field(k))
    return frozenset(keys)


_DA_KEYS = _all_config_keys(DataArrayConfig)
_DS_KEYS = _all_config_keys(DatasetConfig)


class _ClassDescriptor:
    def __init__(self) -> None:
        self.cache: dict[type, Any] = {}


class _FieldsDescriptor(_ClassDescriptor):
    def __get__(
        self, obj: Any, cls: type
    ) -> dict[str, tuple[AnnotationInfo, Any]]:
        if self.cache.get(cls) is None:
            self.cache[cls] = cls._collect_fields()
        return self.cache[cls]


MODEL_CACHE: dict[tuple[type, int], Any] = {}


class _SchemaDescriptor:
    def __get__(self, obj: Any, cls: type) -> Any:
        tid = threading.get_ident()
        key = (cls, tid)
        if MODEL_CACHE.get(key) is None:
            MODEL_CACHE[key] = cls.build_schema_()
        return MODEL_CACHE[key]


class _ChecksDescriptor(_ClassDescriptor):
    def __get__(self, obj: Any, cls: type) -> dict[str, list[Check]]:
        if self.cache.get(cls) is None:
            infos = typing.cast(
                list[FieldCheckInfo],
                cls._collect_check_infos(CHECK_KEY),
            )
            self.cache[cls] = cls._extract_checks(
                infos, field_names=list(cls.__fields__.keys())
            )
        return self.cache[cls]


class _RootCheckDescriptor(_ClassDescriptor):
    def __get__(self, obj: Any, cls: type) -> list[Check]:
        if self.cache.get(cls) is None:
            infos = cls._collect_check_infos(DATAFRAME_CHECK_KEY)
            custom = cls._extract_df_checks(infos)
            reg = _convert_extras_to_checks(
                {} if cls.__extras__ is None else cls.__extras__
            )
            self.cache[cls] = custom + reg
        return self.cache[cls]


class _ParsersDescriptor(_ClassDescriptor):
    def __get__(self, obj: Any, cls: type) -> dict[str, list[Parser]]:
        if self.cache.get(cls) is None:
            infos = typing.cast(
                list[FieldParserInfo],
                cls._collect_parser_infos(PARSER_KEY),
            )
            self.cache[cls] = cls._extract_parsers(
                infos, field_names=list(cls.__fields__.keys())
            )
        return self.cache[cls]


class _RootParsersDescriptor(_ClassDescriptor):
    def __get__(self, obj: Any, cls: type) -> list[Parser]:
        if self.cache.get(cls) is None:
            infos = cls._collect_parser_infos(DATAFRAME_PARSER_KEY)
            self.cache[cls] = cls._extract_df_parsers(infos)
        return self.cache[cls]


TFields = dict[str, tuple[AnnotationInfo, XarrayFieldInfo]]
TChecks = dict[str, list[Check]]
TParsers = dict[str, list[Parser]]


class _XarrayModelBase(BaseModel):
    """Shared machinery for xarray declarative models."""

    Config: type[DataArrayConfig] | type[DatasetConfig] = DataArrayConfig
    __extras__: dict[str, Any] | None = None
    __schema__ = _SchemaDescriptor()
    __config__: Any = None

    __fields__: ClassVar[TFields] = cast(TFields, _FieldsDescriptor())
    __checks__ = cast(TChecks, _ChecksDescriptor())
    __parsers__: TParsers = cast(TParsers, _ParsersDescriptor())
    __root_checks__ = cast(list[Check], _RootCheckDescriptor())
    __root_parsers__: list[Parser] = cast(
        list[Parser], _RootParsersDescriptor()
    )

    @classmethod
    def _get_model_attrs(cls) -> dict[str, Any]:
        bases = inspect.getmro(cls)[:-1]
        attrs: dict[str, Any] = {}
        for base in reversed(bases):
            if (
                issubclass(base, _XarrayModelBase)
                and base is not _XarrayModelBase
            ):
                attrs.update(base.__dict__)
        return attrs

    @classmethod
    def _collect_fields(cls) -> TFields:
        annotations = typing.get_type_hints(cls, include_extras=True)
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

        fields: TFields = {}
        for field_name, annotation in annotations.items():
            if not _is_field(field_name):
                continue
            field = attrs[field_name]
            if not isinstance(field, XarrayFieldInfo):
                raise SchemaInitError(
                    f"'{field_name}' must use pandera.xarray.Field, "
                    f"not {type(field)}."
                )
            fields[field.name] = (AnnotationInfo(annotation), field)
        return fields

    @classmethod
    def _extract_config_options_and_extras(
        cls, config: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        opts: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        config_keys = (
            _DA_KEYS
            if getattr(cls, "_config_keys", None) == "da"
            else _DS_KEYS
        )
        for name, value in vars(config).items():
            if name in config_keys:
                opts[name] = value
            elif _is_field(name):
                extras[name] = value
        return opts, extras

    @classmethod
    def _collect_config_and_extras(
        cls,
    ) -> tuple[type, dict[str, Any]]:
        bases = inspect.getmro(cls)[:-1]
        bases = tuple(
            b
            for b in bases
            if issubclass(b, _XarrayModelBase) and b is not _XarrayModelBase
        )
        root_model, *models = reversed(bases)
        options, extras = cls._extract_config_options_and_extras(
            root_model.Config
        )
        for model in models:
            cfg = getattr(model, _CONFIG_KEY, {})
            o2, e2 = cls._extract_config_options_and_extras(cfg)
            options.update(o2)
            extras.update(e2)
        return type("Config", (cls.Config,), options), extras

    @classmethod
    def _collect_check_infos(cls, key: str) -> list[Any]:
        bases = inspect.getmro(cls)[:-2]
        bases = tuple(b for b in bases if issubclass(b, _XarrayModelBase))
        names: set[str] = set()
        out: list[Any] = []
        for base in bases:
            for attr_name, attr_value in vars(base).items():
                info = getattr(attr_value, key, None)
                if info is None:
                    continue
                if key == CHECK_KEY:
                    if not isinstance(info, FieldCheckInfo):
                        continue
                elif key == DATAFRAME_CHECK_KEY:
                    if not isinstance(info, CheckInfo) or isinstance(
                        info, FieldCheckInfo
                    ):
                        continue
                else:
                    continue
                if attr_name in names:
                    continue
                names.add(attr_name)
                out.append(info)
        return out

    @classmethod
    def _collect_parser_infos(cls, key: str) -> list[Any]:
        bases = inspect.getmro(cls)[:-2]
        bases = tuple(b for b in bases if issubclass(b, _XarrayModelBase))
        names: set[str] = set()
        out: list[Any] = []
        for base in bases:
            for attr_name, attr_value in vars(base).items():
                info = getattr(attr_value, key, None)
                if info is None:
                    continue
                if key == PARSER_KEY:
                    if not isinstance(info, FieldParserInfo):
                        continue
                elif key == DATAFRAME_PARSER_KEY:
                    if not isinstance(info, ParserInfo) or isinstance(
                        info, FieldParserInfo
                    ):
                        continue
                else:
                    continue
                if attr_name in names:
                    continue
                names.add(attr_name)
                out.append(info)
        return out

    @staticmethod
    def _regex_filter(seq: Any, regexps: Any) -> set[str]:
        import re

        matched: set[str] = set()
        for pattern_str in regexps:
            pattern = re.compile(pattern_str)
            matched.update(filter(pattern.match, seq))
        return matched

    @classmethod
    def _extract_checks(
        cls, check_infos: list[FieldCheckInfo], field_names: list[str]
    ) -> dict[str, list[Check]]:
        checks: dict[str, list[Check]] = {}
        for check_info in check_infos:
            fields_set = {
                f.name if isinstance(f, FieldInfo) else f
                for f in check_info.fields
            }
            matched = (
                cls._regex_filter(field_names, fields_set)
                if check_info.regex
                else fields_set
            )
            check_ = check_info.to_check(cls)
            for field in matched:
                if field not in field_names:
                    raise SchemaInitError(
                        f"Check {check_.name} targets unknown field '{field}'."
                    )
                checks.setdefault(field, []).append(check_)
        return checks

    @classmethod
    def _extract_df_checks(cls, check_infos: list[CheckInfo]) -> list[Check]:
        return [ci.to_check(cls) for ci in check_infos]

    @classmethod
    def _extract_parsers(
        cls, parser_infos: list[FieldParserInfo], field_names: list[str]
    ) -> dict[str, list[Parser]]:
        parsers: dict[str, list[Parser]] = {}
        for pi in parser_infos:
            fields_set = {
                f.name if isinstance(f, FieldInfo) else f for f in pi.fields
            }
            matched = (
                cls._regex_filter(field_names, fields_set)
                if pi.regex
                else fields_set
            )
            parser_ = pi.to_parser(cls)
            for field in matched:
                if field not in field_names:
                    raise SchemaInitError(
                        f"Parser {parser_.name} targets unknown field '{field}'."
                    )
                parsers.setdefault(field, []).append(parser_)
        return parsers

    @classmethod
    def _extract_df_parsers(
        cls, parser_infos: list[ParserInfo]
    ) -> list[Parser]:
        return [pi.to_parser(cls) for pi in parser_infos]

    @classmethod
    def to_schema(cls) -> Any:
        return cls.__schema__

    @classmethod
    @docstring_substitution(validate_doc=BaseModel.validate.__doc__)
    def validate(
        cls,
        check_obj: Any,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Any:
        """%(validate_doc)s"""
        return cls.to_schema().validate(
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )


class DataArrayModel(_XarrayModelBase):
    """Declarative schema for a single :class:`xarray.DataArray`.

    Define a required ``data`` field (dtype + optional :func:`Field`) and
    :class:`~pandera.typing.xarray.Coordinate` fields for coordinates. Use
    nested :class:`Config` for array-level options such as ``dims`` and
    ``name``.
    """

    _config_keys = "da"
    Config: type[DataArrayConfig] = DataArrayConfig

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if _CONFIG_KEY in cls.__dict__:
            if not hasattr(cls.Config, "name"):
                cls.Config.name = None
        else:
            cls.Config = type("Config", (cls.Config,), {})
        super().__init_subclass__(**kwargs)
        hints = typing.get_type_hints(cls, include_extras=True)
        for fname in hints:
            if not _is_field(fname):
                continue
            if fname in cls.__dict__:
                continue
            ann = hints[fname]
            if fname == "data":
                field = XarrayField()
                field.__set_name__(cls, fname)
                setattr(cls, fname, field)
                continue
            if _is_coordinate_annotation(ann):
                field = XarrayField()
                field.__set_name__(cls, fname)
                setattr(cls, fname, field)
                continue
            raise SchemaInitError(
                f"DataArrayModel field '{fname}' must be 'data' or "
                "Coordinate[...]; assign pandera.xarray.Field explicitly "
                "if you need a non-default field."
            )
        cls.__config__, cls.__extras__ = cls._collect_config_and_extras()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.validate(*args, **kwargs)

    @classmethod
    def build_schema_(cls) -> DataArraySchema:
        cfg = cls.__config__
        fields = cls.__fields__
        if "data" not in fields:
            raise SchemaInitError("DataArrayModel requires a 'data' field.")
        data_ann, data_fi = fields["data"]
        dtype = cfg.dtype if cfg.dtype is not None else data_ann.arg
        if dtype in (None, typing.Any):
            raise SchemaInitError(
                "Specify dtype on the 'data' field or Config.dtype."
            )
        extra_checks = cls.__checks__.get("data", [])
        data_checks = list(data_fi.checks) + extra_checks
        extra_parsers = cls.__parsers__.get("data", [])
        data_parsers = list(data_fi.parses) + extra_parsers

        dims = data_fi.dims or cfg.dims
        sizes = data_fi.sizes or cfg.sizes
        shape = data_fi.shape or cfg.shape

        coords: dict[str, Coordinate] = {}
        for fname, (ann, fi) in fields.items():
            if fname == "data":
                continue
            if not _is_coordinate_annotation(ann.raw_annotation):
                raise SchemaInitError(
                    f"Unexpected field {fname!r} on DataArrayModel."
                )
            cdtype = _coordinate_dtype_arg(ann.raw_annotation)
            cchecks = list(fi.checks) + cls.__checks__.get(fname, [])
            cparsers = list(fi.parses) + cls.__parsers__.get(fname, [])
            coords[fname] = Coordinate(
                dtype=cdtype,
                checks=cchecks,
                parsers=cparsers,
                nullable=fi.nullable,
                coerce=fi.coerce or cfg.coerce,
                title=fi.title,
                description=fi.description,
            )

        return DataArraySchema(
            dtype=dtype,
            dims=dims,
            sizes=sizes,
            shape=shape,
            coords=coords or None,
            attrs=cfg.attrs,
            name=cfg.name,
            checks=data_checks + cls.__root_checks__,
            parsers=data_parsers + cls.__root_parsers__,
            coerce=cfg.coerce or data_fi.coerce,
            nullable=cfg.nullable or data_fi.nullable,
            strict_coords=cfg.strict_coords,
            strict_attrs=cfg.strict_attrs,
            chunked=cfg.chunked,
            array_type=cfg.array_type,
        )


class DatasetModel(_XarrayModelBase):
    """Declarative schema for an :class:`xarray.Dataset`."""

    _config_keys = "ds"
    Config: type[DatasetConfig] = DatasetConfig

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if _CONFIG_KEY in cls.__dict__:
            if not hasattr(cls.Config, "name"):
                cls.Config.name = None
        else:
            cls.Config = type("Config", (cls.Config,), {})
        super().__init_subclass__(**kwargs)
        hints = typing.get_type_hints(cls, include_extras=True)
        for fname in hints:
            if not _is_field(fname):
                continue
            if fname in cls.__dict__:
                continue
            ann = hints[fname]
            if _is_coordinate_annotation(ann):
                field = XarrayField()
                field.__set_name__(cls, fname)
                setattr(cls, fname, field)
                continue
            ann_info = AnnotationInfo(ann)
            inner = ann_info.arg
            if isinstance(inner, type) and issubclass(inner, DataArrayModel):
                nfi = XarrayFieldInfo(nested_data_array_model=inner)
                nfi.__set_name__(cls, fname)
                setattr(cls, fname, nfi)
                continue
            field = XarrayField()
            field.__set_name__(cls, fname)
            setattr(cls, fname, field)
        cls.__config__, cls.__extras__ = cls._collect_config_and_extras()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.validate(*args, **kwargs)

    @classmethod
    def build_schema_(cls) -> DatasetSchema:
        cfg = cls.__config__
        fields = cls.__fields__
        data_vars: dict[str, DataVar | DataArraySchema] = {}
        coords: dict[str, Coordinate] = {}

        for fname, (ann, fi) in fields.items():
            if _is_coordinate_annotation(ann.raw_annotation):
                cdtype = _coordinate_dtype_arg(ann.raw_annotation)
                cchecks = list(fi.checks) + cls.__checks__.get(fname, [])
                cparsers = list(fi.parses) + cls.__parsers__.get(fname, [])
                coords[fname] = Coordinate(
                    dtype=cdtype,
                    checks=cchecks,
                    parsers=cparsers,
                    nullable=fi.nullable,
                    coerce=fi.coerce,
                    title=fi.title,
                    description=fi.description,
                )
                continue

            if fi.nested_data_array_model is not None:
                data_vars[fi.name] = fi.nested_data_array_model.to_schema()
                continue

            dtype = ann.arg
            if dtype in (None, typing.Any):
                raise SchemaInitError(
                    f"Field {fname!r} needs a dtype or nested DataArrayModel."
                )
            dv_kwargs = fi.to_data_var_kwargs(dtype, optional=ann.optional)
            dv_kwargs["checks"] = dv_kwargs.get(
                "checks", []
            ) + cls.__checks__.get(fname, [])
            dv_kwargs["parsers"] = list(fi.parses) + cls.__parsers__.get(
                fname, []
            )
            data_vars[fi.name] = DataVar(**dv_kwargs)

        return DatasetSchema(
            data_vars=data_vars,
            coords=coords or None,
            dims=cfg.dims,
            sizes=cfg.sizes,
            attrs=None,
            checks=cls.__root_checks__,
            parsers=cls.__root_parsers__,
            strict=cfg.strict,
            strict_coords=cfg.strict_coords,
            strict_attrs=cfg.strict_attrs,
            name=cfg.name,
        )


Field = XarrayField

__all__ = ["DataArrayModel", "DatasetModel", "Field"]
