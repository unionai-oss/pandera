"""Class-based api for pandas models."""

import copy
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

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
from pandera.engines.pandas_engine import Engine
from pandera.errors import SchemaInitError
from pandera.typing import (
    get_index_types,
    get_series_types,
    AnnotationInfo,
    DataFrame,
)
from pandera.utils import docstring_substitution

# if python version is < 3.11, import Self from typing_extensions
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

SchemaIndex = Union[Index, MultiIndex]


class DataFrameModel(_DataFrameModel[pd.DataFrame, DataFrameSchema]):
    """Model of a pandas :class:`~pandera.api.pandas.container.DataFrameSchema`.

    *new in 0.5.0*

    See the :ref:`User Guide <dataframe-models>` for more.
    """

    Config: Type[BaseConfig] = BaseConfig

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
    def _build_columns_index(  # pylint:disable=too-many-locals,too-many-branches
        cls,
        fields: Dict[str, Tuple[AnnotationInfo, FieldInfo]],
        checks: Dict[str, List[Check]],
        parsers: Dict[str, List[Parser]],
        **multiindex_kwargs: Any,
    ) -> Tuple[
        Dict[str, Column],
        Optional[Union[Index, MultiIndex]],
    ]:
        index_count = sum(
            annotation.origin in get_index_types()
            for annotation, _ in fields.values()
        )

        columns: Dict[str, Column] = {}
        indices: List[Index] = []
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
                except TypeError:
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
                    check_name is None
                    and index_count == 1
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
        cls: Type[Self],
        check_obj: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
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
    def empty(cls: Type[Self], *_args) -> DataFrame[Self]:
        """Create an empty DataFrame with the schema of this model."""
        schema = copy.deepcopy(cls.to_schema())
        schema.coerce = True
        empty_df = schema.coerce_dtype(pd.DataFrame(columns=[*schema.columns]))
        return DataFrame[Self](empty_df)


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
