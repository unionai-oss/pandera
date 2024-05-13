"""Class-based api for polars models."""

import inspect
from typing import Dict, List, Tuple, Type

import pandas as pd
import polars as pl

from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model import get_dtype_kwargs
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.model_config import BaseConfig
from pandera.engines import polars_engine as pe
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.typing.polars import Series


class DataFrameModel(_DataFrameModel[pl.LazyFrame, DataFrameSchema]):
    """Model of a polars :class:`~pandera.api.pandas.container.DataFrameSchema`.

    See the :ref:`User Guide <dataframe-models>` for more.
    """

    Config: Type[BaseConfig] = BaseConfig

    @classmethod
    def build_schema_(cls, **kwargs):
        return DataFrameSchema(
            cls._build_columns(cls.__fields__, cls.__checks__),
            checks=cls.__root_checks__,
            **kwargs,
        )

    @classmethod
    def _build_columns(  # pylint:disable=too-many-locals
        cls,
        fields: Dict[str, Tuple[AnnotationInfo, FieldInfo]],
        checks: Dict[str, List[Check]],
    ) -> Dict[str, Column]:

        columns: Dict[str, Column] = {}
        for field_name, (annotation, field) in fields.items():
            field_checks = checks.get(field_name, [])
            field_name = field.name
            check_name = getattr(field, "check_name", None)

            try:
                engine_dtype = pe.Engine.dtype(annotation.raw_annotation)
                if inspect.isclass(annotation.raw_annotation) and issubclass(
                    annotation.raw_annotation, pe.DataType
                ):
                    # use the raw annotation as the dtype if it's a native
                    # pandera polars datatype
                    dtype = annotation.raw_annotation
                else:
                    dtype = engine_dtype.type
            except (TypeError, ValueError) as exc:
                if annotation.metadata:
                    if field.dtype_kwargs:
                        raise TypeError(
                            "Cannot specify redundant 'dtype_kwargs' "
                            + f"for {annotation.raw_annotation}."
                            + "\n Usage Tip: Drop 'typing.Annotated'."
                        ) from exc
                    dtype_kwargs = get_dtype_kwargs(annotation)
                    dtype = annotation.arg(**dtype_kwargs)  # type: ignore
                elif annotation.default_dtype:
                    dtype = annotation.default_dtype
                else:
                    dtype = annotation.arg  # type: ignore

            if (
                annotation.origin is None
                or isinstance(annotation.origin, pl.datatypes.DataTypeClass)
                or annotation.origin is Series
                or dtype
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
                        name=field_name,
                    )
                    if field
                    else {}
                )
                columns[field_name] = Column(**column_kwargs)

            else:
                raise SchemaInitError(
                    f"Invalid annotation '{field_name}: "
                    f"{annotation.raw_annotation}'."
                )

        return columns

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
