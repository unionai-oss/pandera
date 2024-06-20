"""Class-based API for Ibis models."""

import inspect
from typing import Dict, List, Tuple

import ibis.expr.datatypes as dt
import ibis.expr.types as ir

from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model import get_dtype_kwargs
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.ibis.components import Column
from pandera.api.ibis.container import DataFrameSchema
from pandera.engines import ibis_engine
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo


class DataFrameModel(_DataFrameModel[ir.Table, DataFrameSchema]):
    """Definition of a :class:`~pandera.api.ibis.container.DataFrameSchema`.

    *new in 0.1815.0*

    See the :ref:`User Guide <dataframe-models>` for more.
    """

    @classmethod
    def build_schema_(cls, **kwargs) -> DataFrameSchema:
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
                engine_dtype = ibis_engine.Engine.dtype(
                    annotation.raw_annotation
                )
                if inspect.isclass(annotation.raw_annotation) and issubclass(
                    annotation.raw_annotation, ibis_engine.DataType
                ):
                    # use the raw annotation as the dtype if it's a native
                    # pandera Ibis datatype
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
                or isinstance(annotation.origin, dt.DataType)
                # or annotation.origin is Series  # TODO(deepyaman): Implement `Series`.
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
