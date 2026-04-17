"""Class-based API for pandas models."""

import sys
from typing import Any, Optional, Union, cast

import pandas as pd

from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model import get_dtype_kwargs
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.pandas.components import Column, Index, MultiIndex
from pandera.api.pandas.container import DataFrameSchema
from pandera.api.pandas.model_config import BaseConfig
from pandera.api.parsers import Parser
from pandera.engines import numpy_engine, pandas_engine
from pandera.engines.pandas_engine import Engine
from pandera.errors import SchemaInitError
from pandera.typing import (
    AnnotationInfo,
    DataFrame,
    get_index_types,
    get_series_types,
)
from pandera.utils import docstring_substitution

# if python version is < 3.11, import Self from typing_extensions
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

SchemaIndex = Union[Index, MultiIndex]


def _get_nullable_coercion_dtype(
    dtype: Any,
    *,
    nullable: bool,
    coerce: bool,
) -> Any:
    """Use nullable pandas dtypes for nullable fields when coercing."""
    if dtype is None or not nullable or not coerce:
        return dtype

    try:
        pandera_dtype = Engine.dtype(dtype)
    except (TypeError, ValueError):
        return dtype

    nullable_dtype_map = {
        numpy_engine.Bool: pandas_engine.BOOL,
        numpy_engine.Int8: pandas_engine.INT8,
        numpy_engine.Int16: pandas_engine.INT16,
        numpy_engine.Int32: pandas_engine.INT32,
        numpy_engine.Int64: pandas_engine.INT64,
        numpy_engine.UInt8: pandas_engine.UINT8,
        numpy_engine.UInt16: pandas_engine.UINT16,
        numpy_engine.UInt32: pandas_engine.UINT32,
        numpy_engine.UInt64: pandas_engine.UINT64,
    }

    for numpy_dtype, nullable_dtype in nullable_dtype_map.items():
        if isinstance(pandera_dtype, numpy_dtype):
            return nullable_dtype

    return dtype


class DataFrameModel(_DataFrameModel[pd.DataFrame, DataFrameSchema]):
    """Model of a pandas :class:`~pandera.api.pandas.container.DataFrameSchema`.

    *new in 0.5.0*

    See the :ref:`User Guide <dataframe-models>` for more.
    """

    Config: type[BaseConfig] = BaseConfig

    @classmethod
    def build_schema_(cls, **kwargs) -> DataFrameSchema:
        multiindex_kwargs = {
            name[len("multiindex_") :]: value
            for name, value in vars(cls.__config__).items()
            if name.startswith("multiindex_")
        }
        columns, index = cls._build_columns_index(
            cls.__fields__,
            cls.__checks__,
            cls.__parsers__,
            **multiindex_kwargs,
        )
        return DataFrameSchema(
            columns,
            index=index,
            checks=cls.__root_checks__,
            parsers=cls.__root_parsers__,
            **kwargs,
        )

    @classmethod
    def _build_columns_index(
        cls,
        fields: dict[str, tuple[AnnotationInfo, FieldInfo]],
        checks: dict[str, list[Check]],
        parsers: dict[str, list[Parser]],
        **multiindex_kwargs: Any,
    ) -> tuple[
        dict[str, Column],
        Union[Index, MultiIndex] | None,
    ]:
        index_count = sum(
            annotation.origin in get_index_types()
            for annotation, _ in fields.values()
        )

        columns: dict[str, Column] = {}
        indices: list[Index] = []

        for field_name, (annotation, field) in fields.items():
            field_checks = checks.get(field_name, [])
            field_parsers = parsers.get(field_name, [])
            field_name = field.name
            check_name = getattr(field, "check_name", None)

            use_raw_annotation = False
            if annotation.metadata:
                if field.dtype_kwargs:
                    raise TypeError(
                        "Cannot specify redundant 'dtype_kwargs' "
                        + f"for {annotation.raw_annotation}."
                        + "\n Usage Tip: Drop 'typing.Annotated'."
                    )
                if field.dtype_kwargs:
                    raise TypeError(
                        "Cannot specify redundant 'dtype_kwargs' "
                        + f"for {annotation.raw_annotation}."
                        + "\n Usage Tip: Drop 'typing.Annotated'."
                    )
                # Add check for built-in types before attempting signature inspection  
                if annotation.arg in (str, int, float, bool) or (  
                    hasattr(annotation.arg, '__module__') and   
                    annotation.arg.__module__ == 'builtins'  
                ):  
                    # For built-in types, use the type directly without kwargs <-- str
                    dtype = annotation.arg  
                else:  
                    # For parameterized types, extract kwargs and instantiate  
                    dtype_kwargs = get_dtype_kwargs(annotation)  
                    dtype = annotation.arg(**dtype_kwargs)  # type: ignore
            elif annotation.default_dtype:
                dtype = annotation.default_dtype
            else:
                try:
                    # if the raw annotation is accepted by the engine, use it as
                    # the dtype
                    Engine.dtype(annotation.raw_annotation)
                    dtype = annotation.raw_annotation
                    use_raw_annotation = True
                except (TypeError, ValueError):
                    dtype = annotation.arg

            dtype = None if dtype is Any else dtype

            if (
                annotation.is_annotated_type
                or annotation.origin is None
                or use_raw_annotation
                or annotation.origin in get_series_types()
                or annotation.raw_annotation in get_series_types()
            ):
                if check_name is False:
                    raise SchemaInitError(
                        f"'check_name' is not supported for {field_name}."
                    )

                dtype = _get_nullable_coercion_dtype(
                    dtype,
                    nullable=field.nullable,
                    coerce=field.coerce
                    or bool(getattr(cls.__config__, "coerce", False)),
                )

                column_kwargs = (
                    field.column_properties(
                        dtype,
                        required=not annotation.optional,
                        checks=field_checks,
                        parsers=field_parsers,
                        name=field_name,
                    )
                    if field
                    else {}
                )
                columns[field_name] = Column(**column_kwargs)

            elif (
                annotation.origin in get_index_types()
                or annotation.raw_annotation in get_index_types()
            ):
                if annotation.optional:
                    raise SchemaInitError(
                        f"Index '{field_name}' cannot be Optional."
                    )

                if check_name is False or (
                    # default single index
                    check_name is None and index_count == 1
                ):
                    field_name = None  # type:ignore

                index_kwargs = (
                    field.index_properties(
                        dtype,
                        checks=field_checks,
                        name=field_name,
                    )
                    if field
                    else {}
                )
                index = Index(**index_kwargs)
                indices.append(index)
            else:
                raise SchemaInitError(
                    f"Invalid annotation '{field_name}: "
                    f"{annotation.raw_annotation}'"
                )

        return columns, _build_schema_index(indices, **multiindex_kwargs)

    @classmethod
    @docstring_substitution(validate_doc=BaseSchema.validate.__doc__)
    def validate(
        cls: type[Self],
        check_obj: pd.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> DataFrame[Self]:
        """%(validate_doc)s"""
        return cast(
            DataFrame[Self],
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
            {k: v.type if v else None for k, v in schema.dtypes.items()}
        )
        table_schema = pd.io.json.build_table_schema(empty)

        def _field_json_schema(field):
            return {
                "type": "array",
                "items": {
                    "type": (
                        field["type"]
                        if field["type"] != "any" or "extDtype" not in field
                        else field["extDtype"]
                    )
                },
            }

        return {
            "title": schema.name or "pandera.DataFrameSchema",
            "type": "object",
            "properties": {
                field["name"]: _field_json_schema(field)
                for field in table_schema["fields"]
            },
        }

    @classmethod
    def empty(cls: type[Self], *_args) -> DataFrame[Self]:
        """Create an empty DataFrame with the schema of this model."""
        schema = cls.to_schema()

        data = {}
        for col_name, col_schema in schema.columns.items():
            if col_schema.dtype is not None:
                data[col_name] = pd.array([], dtype=col_schema.dtype.type)
            else:
                data[col_name] = pd.array([])

        if isinstance(schema.index, MultiIndex):
            index = pd.MultiIndex.from_arrays(
                [
                    pd.Index(
                        [],
                        dtype=idx.dtype.type
                        if idx.dtype is not None
                        else None,
                        name=idx.name,
                    )
                    for idx in schema.index.indexes
                ]
            )
        elif isinstance(schema.index, Index):
            index = pd.Index(
                [],
                dtype=schema.index.dtype.type
                if schema.index.dtype is not None
                else None,
                name=schema.index.name,
            )
        else:
            index = None

        empty_df = pd.DataFrame(data, index=index)
        return DataFrame[Self](empty_df)


def _build_schema_index(
    indices: list[Index], **multiindex_kwargs: Any
) -> SchemaIndex | None:
    index: SchemaIndex | None = None
    if indices:
        if len(indices) == 1:
            index = indices[0]
        else:
            index = MultiIndex(indices, **multiindex_kwargs)
    return index
