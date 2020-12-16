"""Class-based api"""
import inspect
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
)

import pandas as pd

from . import schema_components
from . import strategies as st
from .checks import Check
from .errors import SchemaInitError
from .model_components import (
    CHECK_KEY,
    DATAFRAME_CHECK_KEY,
    CheckInfo,
    FieldCheckInfo,
    FieldInfo,
)
from .schemas import DataFrameSchema
from .typing import AnnotationInfo, Index, Series

SchemaIndex = Union[schema_components.Index, schema_components.MultiIndex]


_CONFIG_KEY = "Config"


MODEL_CACHE: Dict[Type["SchemaModel"], DataFrameSchema] = {}


class BaseConfig:  # pylint:disable=R0903
    """Define DataFrameSchema-wide options.

    *new in 0.5.0*
    """

    name: Optional[str] = None  #: name of schema
    coerce: bool = False  #: coerce types of all schema components
    strict: bool = False  #: make sure all specified columns are in dataframe
    ordered: bool = False  #: validate columns order
    multiindex_name: Optional[str] = None  #: name of multiindex

    #: coerce types of all MultiIndex components
    multiindex_coerce: bool = False

    #: make sure all specified columns are in MultiIndex
    multiindex_strict: bool = False

    #: validate MultiIndex in order
    multiindex_ordered: bool = True


_config_options = [
    attr for attr in vars(BaseConfig) if not attr.startswith("_")
]


def _extract_config_options(config: Type) -> Dict[str, Any]:
    return {
        name: value
        for name, value in vars(config).items()
        if name in _config_options
    }


class SchemaModel:
    """Definition of a :class:`~pandera.DataFrameSchema`.

    *new in 0.5.0*

    See the :ref:`User Guide <schema_models>` for more.
    """

    Config: Type[BaseConfig] = BaseConfig
    __schema__: Optional[DataFrameSchema] = None
    __config__: Optional[Type[BaseConfig]] = None
    __fields__: Dict[str, Tuple[AnnotationInfo, Optional[FieldInfo]]] = {}
    __checks__: Dict[str, List[Check]] = {}
    __dataframe_checks__: List[Check] = []

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} may not be instantiated.")

    @classmethod
    def to_schema(cls) -> DataFrameSchema:
        """Create :class:`~pandera.DataFrameSchema` from the :class:`.SchemaModel`."""
        if cls in MODEL_CACHE:
            return MODEL_CACHE[cls]

        annotations = cls._collect_field_annotations()
        cls.__fields__ = cls._collect_fields(annotations)
        check_infos = cast(
            List[FieldCheckInfo], cls._collect_check_infos(CHECK_KEY)
        )
        cls.__checks__ = cls._extract_checks(
            check_infos, field_names=list(cls.__fields__.keys())
        )

        df_check_infos = cls._collect_check_infos(DATAFRAME_CHECK_KEY)
        cls.__dataframe_checks__ = cls._extract_df_checks(df_check_infos)

        cls.__config__ = cls._collect_config()
        mi_kwargs = {
            name[len("multiindex_") :]: value
            for name, value in vars(cls.__config__).items()
            if name.startswith("multiindex_")
        }
        columns, index = cls._build_columns_index(
            cls.__fields__, cls.__checks__, **mi_kwargs
        )
        cls.__schema__ = DataFrameSchema(
            columns,
            index=index,
            checks=cls.__dataframe_checks__,  # type: ignore
            coerce=cls.__config__.coerce,
            strict=cls.__config__.strict,
            name=cls.__config__.name,
            ordered=cls.__config__.ordered,
        )
        if cls not in MODEL_CACHE:
            MODEL_CACHE[cls] = cls.__schema__
        return cls.__schema__

    @classmethod
    @pd.util.Substitution(validate_doc=DataFrameSchema.validate.__doc__)
    def validate(
        cls,
        check_obj: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
    ) -> pd.DataFrame:
        """%(validate_doc)s"""
        return cls.to_schema().validate(
            check_obj, head, tail, sample, random_state, lazy
        )

    @classmethod
    @pd.util.Substitution(strategy_doc=DataFrameSchema.strategy.__doc__)
    @st.strategy_import_error
    def strategy(cls, *, size=None):
        """%(strategy_doc)s"""
        return cls.to_schema().strategy(size=size)

    @classmethod
    @pd.util.Substitution(example_doc=DataFrameSchema.strategy.__doc__)
    @st.strategy_import_error
    def example(cls, *, size=None):
        """%(example_doc)s"""
        return cls.to_schema().example(size=size)

    @classmethod
    def _build_columns_index(  # pylint:disable=too-many-locals
        cls,
        fields: Dict[str, Tuple[AnnotationInfo, Optional[FieldInfo]]],
        checks: Dict[str, List[Check]],
        **multiindex_kwargs: Any,
    ) -> Tuple[
        Dict[str, schema_components.Column],
        Optional[Union[schema_components.Index, schema_components.MultiIndex]],
    ]:
        index_count = sum(
            annotation.origin is Index for annotation, _ in fields.values()
        )

        columns: Dict[str, schema_components.Column] = {}
        indices: List[schema_components.Index] = []
        for field_name, (annotation, field) in fields.items():
            field_checks = checks.get(field_name, [])
            field_name = getattr(field, "alias", None) or field_name
            check_name = getattr(field, "check_name", None)

            if annotation.origin is Series:
                col_constructor = (
                    field.to_column if field else schema_components.Column
                )

                if check_name is False:
                    raise SchemaInitError(
                        f"'check_name' is not supported for {field_name}."
                    )

                columns[field_name] = col_constructor(  # type: ignore
                    annotation.arg,
                    required=not annotation.optional,
                    checks=field_checks,
                    name=field_name,
                )
            elif annotation.origin is Index:
                if annotation.optional:
                    raise SchemaInitError(
                        f"Index '{field_name}' cannot be Optional."
                    )

                if check_name is False or (
                    # default single index
                    check_name is None
                    and index_count == 1
                ):
                    field_name = None  # type:ignore

                index_constructor = (
                    field.to_index if field else schema_components.Index
                )
                index = index_constructor(  # type: ignore
                    annotation.arg, checks=field_checks, name=field_name
                )
                indices.append(index)
            else:
                raise SchemaInitError(
                    f"Invalid annotation '{field_name}: {annotation.raw_annotation}'"
                )

        return columns, _build_schema_index(indices, **multiindex_kwargs)

    @classmethod
    def _collect_field_annotations(cls) -> Dict[str, AnnotationInfo]:
        """Collect inherited field annotations from bases."""
        bases = inspect.getmro(cls)[:-2]  # bases -> SchemaModel -> object
        bases = cast(Tuple[Type[SchemaModel]], bases)
        raw_annotations = {}
        for base in reversed(bases):
            base_annotations = _get_field_annotations(base)
            raw_annotations.update(base_annotations)
        return {
            name: AnnotationInfo(annotation)
            for name, annotation in raw_annotations.items()
        }

    @classmethod
    def _collect_fields(
        cls, annotations: Dict[str, AnnotationInfo]
    ) -> Dict[str, Tuple[AnnotationInfo, Optional[FieldInfo]]]:
        """Centralize publicly named fields and their corresponding annotations."""
        fields = {}
        for field_name, annotation in annotations.items():
            field: Optional[FieldInfo] = getattr(cls, field_name, None)
            if field is not None and not isinstance(field, FieldInfo):
                raise SchemaInitError(
                    f"'{field_name}' can only be assigned a 'Field', "
                    + f"not a '{type(field)}.'"
                )
            field_name = getattr(field, "alias", None) or field_name
            fields[field_name] = (annotation, field)
        return fields

    @classmethod
    def _collect_config(cls) -> Type[BaseConfig]:
        """Collect config options from bases."""
        bases = inspect.getmro(cls)[:-1]
        bases = cast(Tuple[Type[SchemaModel]], bases)
        root_model, *models = reversed(bases)

        options = _extract_config_options(root_model.Config)
        for model in models:
            config = getattr(model, _CONFIG_KEY, {})
            base_options = _extract_config_options(config)
            options.update(base_options)
        return type("Config", (BaseConfig,), options)

    @classmethod
    def _collect_check_infos(cls, key: str) -> List[CheckInfo]:
        """Collect inherited check metadata from bases.
        Inherited classmethods are not in cls.__dict__, that's why we need to
        walk the inheritance tree.
        """
        bases = inspect.getmro(cls)[:-2]  # bases -> SchemaModel -> object
        bases = cast(Tuple[Type[SchemaModel]], bases)

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
            if check_info.regex:
                matched = _regex_filter(field_names, check_info.fields)
            else:
                matched = check_info.fields

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


def _get_field_annotations(model: Type[SchemaModel]) -> Dict[str, Any]:
    annotations = get_type_hints(model)

    def _not_routine(member: Any) -> bool:
        return not inspect.isroutine(member)

    missing = []
    for name, _ in inspect.getmembers(model, _not_routine):
        if name.startswith("_") or name == _CONFIG_KEY:
            annotations.pop(name, None)
        elif name not in annotations:
            missing.append(name)

    if missing:
        raise SchemaInitError(f"Found missing annotations: {missing}")

    return annotations


def _build_schema_index(
    indices: List[schema_components.Index], **multiindex_kwargs: Any
) -> Optional[SchemaIndex]:
    index: Optional[SchemaIndex] = None
    if indices:
        if len(indices) == 1:
            index = indices[0]
        else:
            index = schema_components.MultiIndex(indices, **multiindex_kwargs)
    return index


def _regex_filter(seq: Iterable, regexps: Iterable[str]) -> Set[str]:
    """Filter items matching at least one of the regexes."""
    matched: Set[str] = set()
    for regex in regexps:
        pattern = re.compile(regex)
        matched.update(filter(pattern.match, seq))
    return matched
