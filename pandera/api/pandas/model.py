"""Class-based api for pandas models."""

import inspect
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import pandas as pd
from pandera.api.checks import Check
from pandera.api.dataframe.model import (
    DataFrameModel as _DataFrameModel,
    TDataFrameModel,
)
from pandera.api.pandas.container import DataFrameSchema
from pandera.api.pandas.components import Column, Index, MultiIndex
from pandera.api.pandas.model_components import FieldInfo
from pandera.api.pandas.model_config import BaseConfig
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo, INDEX_TYPES, SERIES_TYPES
from pandera.typing.common import DataFrameBase
from pandera.utils import docstring_substitution


SchemaIndex = Union[Index, MultiIndex]


class DataFrameModel(_DataFrameModel):
    """Definition of a :class:`~pandera.api.pandas.container.DataFrameSchema`.

    *new in 0.5.0*

    .. important::

        This class is the new name for ``SchemaModel``, which will be deprecated
        in pandera version ``0.20.0``.

    See the :ref:`User Guide <dataframe_models>` for more.
    """

    Config: Type[BaseConfig] = BaseConfig
    __field_info_cls__ = FieldInfo

    @docstring_substitution(validate_doc=DataFrameSchema.validate.__doc__)
    def __new__(cls, *args, **kwargs) -> DataFrameBase[TDataFrameModel]:  # type: ignore [misc]
        """%(validate_doc)s"""
        return cast(
            DataFrameBase[TDataFrameModel], cls.validate(*args, **kwargs)
        )

    @classmethod
    def build_schema_(cls, **kwargs):
        multiindex_kwargs = {
            name[len("multiindex_") :]: value
            for name, value in vars(cls.__config__).items()
            if name.startswith("multiindex_")
        }
        columns, index = cls._build_columns_index(
            cls.__fields__, cls.__checks__, **multiindex_kwargs
        )
        return DataFrameSchema(
            columns,
            index=index,
            checks=cls.__root_checks__,
            **kwargs,
        )

    @classmethod
    def _build_columns_index(  # pylint:disable=too-many-locals
        cls,
        fields: Dict[str, Tuple[AnnotationInfo, FieldInfo]],
        checks: Dict[str, List[Check]],
        **multiindex_kwargs: Any,
    ) -> Tuple[Dict[str, Column], Optional[Union[Index, MultiIndex]],]:
        index_count = sum(
            annotation.origin in INDEX_TYPES
            for annotation, _ in fields.values()
        )

        columns: Dict[str, Column] = {}
        indices: List[Index] = []
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

            if (
                annotation.origin is None
                or annotation.origin in SERIES_TYPES
                or annotation.raw_annotation in SERIES_TYPES
            ):
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
            elif (
                annotation.origin in INDEX_TYPES
                or annotation.raw_annotation in INDEX_TYPES
            ):
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

                index_constructor = field.to_index if field else Index
                index = index_constructor(  # type: ignore
                    dtype, checks=field_checks, name=field_name
                )
                indices.append(index)
            else:
                raise SchemaInitError(
                    f"Invalid annotation '{field_name}: "
                    f"{annotation.raw_annotation}'"
                )

        return columns, _build_schema_index(indices, **multiindex_kwargs)

    @classmethod
    def to_yaml(cls, stream: Optional[os.PathLike] = None):
        """
        Convert `Schema` to yaml using `io.to_yaml`.
        """
        return cls.to_schema().to_yaml(stream)

    @classmethod
    @docstring_substitution(validate_doc=DataFrameSchema.validate.__doc__)
    def validate(
        cls: Type[TDataFrameModel],
        check_obj: pd.DataFrame,
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
    def to_json_schema(cls):
        """Serialize schema metadata into json-schema format.

        :param dataframe_schema: schema to write to json-schema format.

        .. note::

            This function is currently does not fully specify a pandera schema,
            and is primarily used internally to render OpenAPI docs via the
            FastAPI integration.
        """
        schema = cls.to_schema()
        empty = pd.DataFrame(columns=schema.columns.keys()).astype(
            {k: v.type for k, v in schema.dtypes.items()}
        )
        table_schema = pd.io.json.build_table_schema(empty)

        def _field_json_schema(field):
            return {
                "type": "array",
                "items": {"type": field["type"]},
            }

        return {
            "title": schema.name or "pandera.DataFrameSchema",
            "type": "object",
            "properties": {
                field["name"]: _field_json_schema(field)
                for field in table_schema["fields"]
            },
        }


SchemaModel = DataFrameModel
"""
Alias for DataFrameModel.

.. warning::

   This subclass is necessary for backwards compatibility, and will be
   deprecated in pandera version ``0.20.0`` in favor of
   :py:class:`~pandera.api.pandas.model.DataFrameModel`
"""


def _build_schema_index(
    indices: List[Index], **multiindex_kwargs: Any
) -> Optional[SchemaIndex]:
    index: Optional[SchemaIndex] = None
    if indices:
        if len(indices) == 1:
            index = indices[0]
        else:
            index = MultiIndex(indices, **multiindex_kwargs)
    return index


def _get_dtype_kwargs(annotation: AnnotationInfo) -> Dict[str, Any]:
    sig = inspect.signature(annotation.arg)  # type: ignore
    dtype_arg_names = list(sig.parameters.keys())
    if len(annotation.metadata) != len(dtype_arg_names):  # type: ignore
        raise TypeError(
            f"Annotation '{annotation.arg.__name__}' requires "  # type: ignore
            + f"all positional arguments {dtype_arg_names}."
        )
    return dict(zip(dtype_arg_names, annotation.metadata))  # type: ignore
