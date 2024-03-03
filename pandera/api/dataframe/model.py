"""Class-based api for pandas models."""

import copy
import inspect
import os
import re
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pandera.api.base.model import BaseModel
from pandera.api.checks import Check
from pandera.api.base.schema import BaseSchema
from pandera.api.dataframe.model_components import (
    CHECK_KEY,
    DATAFRAME_CHECK_KEY,
    CheckInfo,
    Field,
    FieldCheckInfo,
    FieldInfo,
)
from pandera.api.dataframe.model_config import BaseConfig
from pandera.engines import PYDANTIC_V2
from pandera.errors import SchemaInitError
from pandera.strategies import base_strategies as st
from pandera.typing import AnnotationInfo
from pandera.typing.common import DataFrameBase

if PYDANTIC_V2:
    from pydantic_core import core_schema
    from pydantic import GetJsonSchemaHandler, GetCoreSchemaHandler

try:
    from typing_extensions import get_type_hints
except ImportError:  # pragma: no cover
    from typing import get_type_hints  # type: ignore


F = TypeVar("F", bound=Callable)
TDataFrame = TypeVar("TDataFrame")
TDataFrameModel = TypeVar("TDataFrameModel", bound="DataFrameModel")
TSchema = TypeVar("TSchema", bound=BaseSchema)

_CONFIG_KEY = "Config"
MODEL_CACHE: Dict[Type["DataFrameModel"], Any] = {}
GENERIC_SCHEMA_CACHE: Dict[
    Tuple[Type["DataFrameModel"], Tuple[Type[Any], ...]],
    Type["DataFrameModel"],
] = {}


def docstring_substitution(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper around pandas.util.Substitution."""

    def decorator(func: F) -> F:
        if args is not None:
            _doc = func.__doc__ % tuple(args)  # type: ignore[operator]
        elif kwargs:
            _doc = func.__doc__ % kwargs
        func.__doc__ = _doc
        return func

    return decorator


def _is_field(name: str) -> bool:
    """Ignore private and reserved keywords."""
    return not name.startswith("_") and name != _CONFIG_KEY


def _convert_extras_to_checks(extras: Dict[str, Any]) -> List[Check]:
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


class DataFrameModel(BaseModel, Generic[TDataFrame, TSchema]):
    """Definition of a generic DataFrame model.

    .. important::

        This class is the new name for ``SchemaModel``, which will be deprecated
        in pandera version ``0.20.0``.

    See the :ref:`User Guide <dataframe_models>` for more.
    """

    Config: Type[BaseConfig] = BaseConfig
    __extras__: Optional[Dict[str, Any]] = None
    __schema__: Optional[TSchema] = None
    __config__: Optional[Type[BaseConfig]] = None

    #: Key according to `FieldInfo.name`
    __fields__: Mapping[str, Tuple[AnnotationInfo, FieldInfo]] = {}
    __checks__: Dict[str, List[Check]] = {}
    __root_checks__: List[Check] = []

    def __new__(cls, *args, **kwargs) -> DataFrameBase[TDataFrameModel]:  # type: ignore [misc]
        """%(validate_doc)s"""
        return cast(
            DataFrameBase[TDataFrameModel], cls.validate(*args, **kwargs)
        )

    def __init_subclass__(cls, **kwargs):
        """Ensure :class:`~pandera.api.pandas.model_components.FieldInfo` instances."""
        if "Config" in cls.__dict__:
            cls.Config.name = (
                cls.Config.name
                if hasattr(cls.Config, "name")
                else cls.__name__
            )
        else:
            cls.Config = type("Config", (cls.Config,), {"name": cls.__name__})

        super().__init_subclass__(**kwargs)
        # pylint:disable=no-member
        subclass_annotations = cls.__dict__.get("__annotations__", {})
        for field_name in subclass_annotations.keys():
            if _is_field(field_name) and field_name not in cls.__dict__:
                # Field omitted
                field = Field()
                field.__set_name__(cls, field_name)
                setattr(cls, field_name, field)

        cls.__config__, cls.__extras__ = cls._collect_config_and_extras()

    def __class_getitem__(
        cls: Type[TDataFrameModel],
        params: Union[Type[Any], Tuple[Type[Any], ...]],
    ) -> Type[TDataFrameModel]:
        """Parameterize the class's generic arguments with the specified types"""
        if not hasattr(cls, "__parameters__"):
            raise TypeError(
                f"{cls.__name__} must inherit from typing.Generic before being parameterized"
            )
        # pylint: disable=no-member
        __parameters__: Tuple[TypeVar, ...] = cls.__parameters__  # type: ignore

        if not isinstance(params, tuple):
            params = (params,)
        if len(params) != len(__parameters__):
            raise ValueError(
                f"Expected {len(__parameters__)} generic arguments but found {len(params)}"
            )
        if (cls, params) in GENERIC_SCHEMA_CACHE:
            return typing.cast(
                Type[TDataFrameModel], GENERIC_SCHEMA_CACHE[(cls, params)]
            )

        param_dict: Dict[TypeVar, Type[Any]] = dict(
            zip(__parameters__, params)
        )
        extra: Dict[str, Any] = {"__annotations__": {}}
        for field, (annot_info, field_info) in cls._collect_fields().items():
            if isinstance(annot_info.arg, TypeVar):
                if annot_info.arg in param_dict:
                    raw_annot = annot_info.origin[param_dict[annot_info.arg]]  # type: ignore
                    if annot_info.optional:
                        raw_annot = Optional[raw_annot]
                    extra["__annotations__"][field] = raw_annot
                    extra[field] = copy.deepcopy(field_info)

        parameterized_name = (
            f"{cls.__name__}[{', '.join(p.__name__ for p in params)}]"
        )
        parameterized_cls = type(parameterized_name, (cls,), extra)
        GENERIC_SCHEMA_CACHE[(cls, params)] = parameterized_cls
        return parameterized_cls

    @classmethod
    def _build_schema(cls, **kwargs) -> TSchema:
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> TSchema:
        """Create :class:`~pandera.DataFrameSchema` from the :class:`.DataFrameModel`."""
        if cls in MODEL_CACHE:
            return MODEL_CACHE[cls]

        cls.__fields__ = cls._collect_fields()
        for field, (annot_info, _) in cls.__fields__.items():
            if isinstance(annot_info.arg, TypeVar):
                raise SchemaInitError(f"Field {field} has a generic data type")

        check_infos = typing.cast(
            List[FieldCheckInfo], cls._collect_check_infos(CHECK_KEY)
        )

        cls.__checks__ = cls._extract_checks(
            check_infos, field_names=list(cls.__fields__.keys())
        )

        df_check_infos = cls._collect_check_infos(DATAFRAME_CHECK_KEY)
        df_custom_checks = cls._extract_df_checks(df_check_infos)
        df_registered_checks = _convert_extras_to_checks(
            {} if cls.__extras__ is None else cls.__extras__
        )
        cls.__root_checks__ = df_custom_checks + df_registered_checks

        kwargs = {}
        if cls.__config__ is not None:
            kwargs = {
                "dtype": cls.__config__.dtype,
                "coerce": cls.__config__.coerce,
                "strict": cls.__config__.strict,
                "name": cls.__config__.name,
                "ordered": cls.__config__.ordered,
                "unique": cls.__config__.unique,
                "title": cls.__config__.title,
                "description": cls.__config__.description or cls.__doc__,
                "unique_column_names": cls.__config__.unique_column_names,
                "add_missing_columns": cls.__config__.add_missing_columns,
                "drop_invalid_rows": cls.__config__.drop_invalid_rows,
            }
        cls.__schema__ = cls._build_schema(**kwargs)
        if cls not in MODEL_CACHE:
            MODEL_CACHE[cls] = cls.__schema__  # type: ignore
        return cls.__schema__  # type: ignore

    @classmethod
    def to_yaml(cls, stream: Optional[os.PathLike] = None):
        """
        Convert `Schema` to yaml using `io.to_yaml`.
        """
        return cls.to_schema().to_yaml(stream)

    @classmethod
    def validate(
        cls: Type[TDataFrameModel],
        check_obj: TDataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> DataFrameBase[TDataFrameModel]:
        """%(validate_doc)s"""
        return cast(
            DataFrameBase[TDataFrameModel],
            cls.to_schema().validate(
                check_obj, head, tail, sample, random_state, lazy, inplace
            ),
        )

    @classmethod
    @st.strategy_import_error
    def strategy(cls: Type[TDataFrameModel], **kwargs):
        """%(strategy_doc)s"""
        return cls.to_schema().strategy(**kwargs)

    @classmethod
    @st.strategy_import_error
    def example(
        cls: Type[TDataFrameModel],
        **kwargs,
    ) -> DataFrameBase[TDataFrameModel]:
        """%(example_doc)s"""
        return cast(
            DataFrameBase[TDataFrameModel], cls.to_schema().example(**kwargs)
        )

    @classmethod
    def _get_model_attrs(cls) -> Dict[str, Any]:
        """Return all attributes.
        Similar to inspect.get_members but bypass descriptors __get__.
        """
        bases = inspect.getmro(cls)[:-1]  # bases -> DataFrameModel -> object
        attrs = {}
        for base in reversed(bases):
            if issubclass(base, DataFrameModel):
                attrs.update(base.__dict__)
        return attrs

    @classmethod
    def _collect_fields(cls) -> Dict[str, Tuple[AnnotationInfo, FieldInfo]]:
        """Centralize publicly named fields and their corresponding annotations."""
        # pylint: disable=unexpected-keyword-arg
        annotations = get_type_hints(  # type: ignore[call-arg]
            cls,
            include_extras=True,
        )
        # pylint: enable=unexpected-keyword-arg
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
            field = attrs[field_name]  # __init_subclass__ guarantees existence
            if not isinstance(field, FieldInfo):
                raise SchemaInitError(
                    f"'{field_name}' can only be assigned a 'Field', "
                    + f"not a '{type(field)}.'"
                )
            fields[field.name] = (AnnotationInfo(annotation), field)
        return fields

    @classmethod
    def _extract_config_options_and_extras(
        cls,
        config: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_options, extras = {}, {}
        _config_options = [
            attr for attr in vars(cls.Config) if _is_field(attr)
        ]
        for name, value in vars(config).items():
            if name in _config_options:
                config_options[name] = value
            elif _is_field(name):
                extras[name] = value
            # drop private/reserved keywords

        return config_options, extras

    @classmethod
    def _collect_config_and_extras(
        cls,
    ) -> Tuple[Type[BaseConfig], Dict[str, Any]]:
        """Collect config options from bases, splitting off unknown options."""
        bases = inspect.getmro(cls)[:-1]
        bases = tuple(
            base for base in bases if issubclass(base, DataFrameModel)
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
    def _collect_check_infos(cls, key: str) -> List[CheckInfo]:
        """Collect inherited check metadata from bases.
        Inherited classmethods are not in cls.__dict__, that's why we need to
        walk the inheritance tree.
        """
        bases = inspect.getmro(cls)[:-2]  # bases -> DataFrameModel -> object
        bases = tuple(
            base for base in bases if issubclass(base, DataFrameModel)
        )

        method_names = set()
        check_infos = []
        for base in bases:
            for attr_name, attr_value in vars(base).items():
                check_info = getattr(attr_value, key, None)
                if not isinstance(check_info, CheckInfo):
                    continue
                if attr_name in method_names:  # check overridden by subclass
                    continue
                method_names.add(attr_name)
                check_infos.append(check_info)
        return check_infos

    @staticmethod
    def _regex_filter(seq: Iterable, regexps: Iterable[str]) -> Set[str]:
        """Filter items matching at least one of the regexes."""
        matched: Set[str] = set()
        for regex in regexps:
            pattern = re.compile(regex)
            matched.update(filter(pattern.match, seq))
        return matched

    @classmethod
    def _extract_checks(
        cls, check_infos: List[FieldCheckInfo], field_names: List[str]
    ) -> Dict[str, List[Check]]:
        """Collect field annotations from bases in mro reverse order."""
        checks: Dict[str, List[Check]] = {}
        for check_info in check_infos:
            check_info_fields = {
                field.name if isinstance(field, FieldInfo) else field
                for field in check_info.fields
            }
            if check_info.regex:
                matched = cls._regex_filter(field_names, check_info_fields)
            else:
                matched = check_info_fields

            check_ = check_info.to_check(cls)

            for field in matched:
                if field not in field_names:
                    raise SchemaInitError(
                        f"Check {check_.name} is assigned to a non-existing field '{field}'."
                    )
                if field not in checks:
                    checks[field] = []
                checks[field].append(check_)
        return checks

    @classmethod
    def _extract_df_checks(cls, check_infos: List[CheckInfo]) -> List[Check]:
        """Collect field annotations from bases in mro reverse order."""
        return [check_info.to_check(cls) for check_info in check_infos]

    @classmethod
    def get_metadata(cls) -> Optional[dict]:
        """Provide metadata for columns and schema level"""
        res: Dict[Any, Any] = {"columns": {}}
        columns = cls._collect_fields()

        for k, (_, v) in columns.items():
            res["columns"][k] = v.properties["metadata"]

        res["dataframe"] = cls.Config.metadata

        meta = {}
        meta[cls.Config.name] = res
        return meta

    @classmethod
    def pydantic_validate(cls, schema_model: Any) -> "DataFrameModel":
        """Verify that the input is a compatible dataframe model."""
        if not inspect.isclass(schema_model):  # type: ignore
            raise TypeError(f"{schema_model} is not a pandera.DataFrameModel")

        if not issubclass(schema_model, cls):  # type: ignore
            raise TypeError(f"{schema_model} does not inherit {cls}.")

        try:
            schema_model.to_schema()
        except SchemaInitError as exc:
            raise ValueError(
                f"Cannot use {cls} as a pydantic type as its "
                "DataFrameModel cannot be converted to a DataFrameSchema.\n"
                f"Please revisit the model to address the following errors:"
                f"\n{exc}"
            ) from exc

        return cast("DataFrameModel", schema_model)

    @classmethod
    def _to_json_schema(cls):
        """Serialize schema metadata into json-schema format."""
        raise NotImplementedError

    if PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls.pydantic_validate,
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls,
            _core_schema: core_schema.CoreSchema,
            _handler: GetJsonSchemaHandler,
        ):
            """Update pydantic field schema."""
            json_schema = _handler(_core_schema)
            json_schema = _handler.resolve_ref_schema(json_schema)
            json_schema.update(cls._to_json_schema())

    else:

        @classmethod
        def __modify_schema__(cls, field_schema):
            """Update pydantic field schema."""
            field_schema.update(cls._to_json_schema())

        @classmethod
        def __get_validators__(cls):
            yield cls.pydantic_validate
