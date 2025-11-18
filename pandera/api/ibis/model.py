"""Class-based API for Ibis models."""

import inspect
import sys
from typing import (
    Optional,
    cast,
)

import ibis
import ibis.expr.datatypes as dt

from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model import get_dtype_kwargs
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.ibis.components import Column
from pandera.api.ibis.container import DataFrameSchema
from pandera.engines import ibis_engine
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.typing.ibis import Table
from pandera.utils import docstring_substitution

# if python version is < 3.11, import Self from typing_extensions
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class DataFrameModel(_DataFrameModel[ibis.Table, DataFrameSchema]):
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
    def _build_columns(
        cls,
        fields: dict[str, tuple[AnnotationInfo, FieldInfo]],
        checks: dict[str, list[Check]],
    ) -> dict[str, Column]:
        columns: dict[str, Column] = {}
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

    @classmethod
    @docstring_substitution(validate_doc=BaseSchema.validate.__doc__)
    def validate(
        cls: type[Self],
        check_obj: ibis.Table,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Table[Self]:
        """%(validate_doc)s"""
        result = cls.to_schema().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )
        return cast(Table[Self], result)
