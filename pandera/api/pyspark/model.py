"""Class-based api for pyspark models."""

# pylint:disable=abstract-method
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

import pyspark.sql as ps
from pyspark.sql.types import StructType

from pandera.api.base.model import BaseModel
from pandera.api.checks import Check
from pandera.api.pyspark.components import Column
from pandera.api.pyspark.container import DataFrameSchema
from pandera.api.pyspark.model_components import (
    CHECK_KEY,
    DATAFRAME_CHECK_KEY,
    CheckInfo,
    Field,
    FieldCheckInfo,
    FieldInfo,
)
from pandera.api.pyspark.model_config import BaseConfig
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.typing.common import DataFrameBase

try:
    from typing_extensions import get_type_hints
except ImportError:  # pragma: no cover
    from typing import get_type_hints  # type: ignore


_CONFIG_KEY = "Config"
MODEL_CACHE: Dict[Type["DataFrameModel"], DataFrameSchema] = {}
GENERIC_SCHEMA_CACHE: Dict[
    Tuple[Type["DataFrameModel"], Tuple[Type[Any], ...]],
    Type["DataFrameModel"],
] = {}

F = TypeVar("F", bound=Callable)
TDataFrameModel = TypeVar("TDataFrameModel", bound="DataFrameModel")


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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config_options, extras = {}, {}
    for name, value in vars(config).items():
        if name in _config_options:
            config_options[name] = value
        elif _is_field(name):
            extras[name] = value
        # drop private/reserved keywords

    return config_options, extras


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


class DataFrameModel(BaseModel):
    """Definition of a :class:`~pandera.api.pyspark.container.DataFrameSchema`.

    *new in 0.16.0*


    See the :ref:`User Guide <dataframe-models>` for more.
    """

    Config: Type[BaseConfig] = BaseConfig
    __extras__: Optional[Dict[str, Any]] = None
    __schema__: Optional[DataFrameSchema] = None
    __config__: Optional[Type[BaseConfig]] = None

    #: Key according to `FieldInfo.name`
    __fields__: Mapping[str, Tuple[AnnotationInfo, FieldInfo]] = {}
    __checks__: Dict[str, List[Check]] = {}
    __root_checks__: List[Check] = []

    @docstring_substitution(validate_doc=DataFrameSchema.validate.__doc__)
    def __new__(cls, *args, **kwargs) -> DataFrameBase[TDataFrameModel]:  # type: ignore [misc]
        """%(validate_doc)s"""
        return cast(
            DataFrameBase[TDataFrameModel], cls.validate(*args, **kwargs)
        )

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
    def to_schema(cls) -> DataFrameSchema:
        """Create :class:`~pandera.pyspark.DataFrameSchema` from the :class:`.DataFrameModel`."""

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

        columns = cls._build_columns_index(cls.__fields__, cls.__checks__)

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
            }
        cls.__schema__ = DataFrameSchema(
            columns,
            checks=cls.__root_checks__,  # type: ignore
            **kwargs,  # type: ignore
        )

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
    def to_structtype(cls) -> StructType:
        """Recover fields of DataFrameModel as a Pyspark StructType object.

        :returns: StructType object with current model fields.
        """
        return cls.to_schema().to_structtype()

    @classmethod
    def to_ddl(cls) -> str:
        """Recover fields of DataFrameModel as a Pyspark DDL string.

        :returns: String with current model fields, in compact DDL format.
        """
        return cls.to_schema().to_ddl()

    @classmethod
    @docstring_substitution(validate_doc=DataFrameSchema.validate.__doc__)
    def validate(
        cls: Type[TDataFrameModel],
        check_obj: ps.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = True,
        inplace: bool = False,
    ) -> Optional[DataFrameBase[TDataFrameModel]]:
        """%(validate_doc)s"""
        return cast(
            DataFrameBase[TDataFrameModel],
            cls.to_schema().validate(
                check_obj, head, tail, sample, random_state, lazy, inplace
            ),
        )

    @classmethod
    def _build_columns_index(  # pylint:disable=too-many-locals
        cls,
        fields: Dict[str, Tuple[AnnotationInfo, FieldInfo]],
        checks: Dict[str, List[Check]],
    ) -> Dict[str, Column]:
        columns: Dict[str, Column] = {}
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
                col_constructor = field.to_column if field else Column

                if check_name is False:
                    raise SchemaInitError(
                        f"'check_name' is not supported for {field_name}."
                    )

                columns[field_name] = col_constructor(  # type: ignore
                    dtype,
                    required=not annotation.optional,
                    checks=field_checks,
                    name=field_name,
                )
            else:
                raise SchemaInitError(
                    f"Invalid annotation '{field_name}: "
                    f"{annotation.raw_annotation}'"
                )

        return columns

    @classmethod
    def _get_model_attrs(cls) -> Dict[str, Any]:
        """Return all attributes.
        Similar to inspect.get_members but bypass descriptors __get__.
        """
        bases = inspect.getmro(cls)[:-1]  # bases -> DataFrameModel -> object
        attrs: Dict[str, Any] = {}
        for base in reversed(bases):
            if issubclass(base, DataFrameModel):
                attrs.update(base.__dict__)
        return attrs

    @classmethod
    def _collect_fields(cls) -> Dict[str, Tuple[AnnotationInfo, FieldInfo]]:
        """Centralize publicly named fields and their corresponding annotations."""

        annotations = get_type_hints(  # pylint:disable=unexpected-keyword-arg
            cls, include_extras=True  # type: ignore [call-arg]
        )
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
                    + f"not a '{type(field)}'."
                )
            fields[field.name] = (AnnotationInfo(annotation), field)

        return fields

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

        options, extras = _extract_config_options_and_extras(root_model.Config)

        for model in models:
            config = getattr(model, _CONFIG_KEY, {})
            base_options, base_extras = _extract_config_options_and_extras(
                config
            )
            options.update(base_options)
            extras.update(base_extras)

        return type("Config", (BaseConfig,), options), extras

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
                matched = _regex_filter(field_names, check_info_fields)
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
    def __get_validators__(cls):
        yield cls.pydantic_validate

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


def _regex_filter(seq: Iterable, regexps: Iterable[str]) -> Set[str]:
    """Filter items matching at least one of the regexes."""
    matched: Set[str] = set()
    for regex in regexps:
        pattern = re.compile(regex)
        matched.update(filter(pattern.match, seq))
    return matched


def _get_dtype_kwargs(annotation: AnnotationInfo) -> Dict[str, Any]:
    sig = inspect.signature(annotation.arg)  # type: ignore
    dtype_arg_names = list(sig.parameters.keys())
    if len(annotation.metadata) != len(dtype_arg_names):  # type: ignore
        raise TypeError(
            f"Annotation '{annotation.arg.__name__}' requires "  # type: ignore
            + f"all positional arguments {dtype_arg_names}."
        )
    return dict(zip(dtype_arg_names, annotation.metadata))  # type: ignore
