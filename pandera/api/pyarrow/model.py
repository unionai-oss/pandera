"""Class-based API for PyArrow models."""

from __future__ import annotations

import inspect
import sys
from typing import cast

import pyarrow as pa

from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model import (
    _dtype_metadata,
    get_dtype_kwargs,
)
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.pyarrow.components import Column
from pandera.api.pyarrow.container import DataFrameSchema
from pandera.api.pyarrow.model_config import BaseConfig
from pandera.api.pyarrow.utils import resolve_dtype
from pandera.engines import narwhals_engine
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.typing.pyarrow import Table
from pandera.utils import docstring_substitution

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class DataFrameModel(_DataFrameModel[pa.Table, DataFrameSchema]):
    """Definition of a :class:`~pandera.api.pyarrow.container.DataFrameSchema`.

    See the :ref:`User Guide <dataframe-models>` for more.
    """

    Config: type[BaseConfig] = BaseConfig

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
                engine_dtype = resolve_dtype(annotation.raw_annotation)
                if inspect.isclass(annotation.raw_annotation) and issubclass(
                    annotation.raw_annotation, narwhals_engine.DataType
                ):
                    # use the raw annotation as the dtype if it's a native
                    # pandera narwhals datatype
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
                    # ``Annotated`` may carry only a FieldInfo (e.g.
                    # ``Annotated[float, pa.Field(...)]``) without any
                    # dtype parameters. In that case, use the annotated
                    # type as-is.
                    if _dtype_metadata(annotation):
                        dtype_kwargs = get_dtype_kwargs(annotation)
                        dtype = annotation.arg(**dtype_kwargs)  # type: ignore
                    else:
                        dtype = annotation.arg  # type: ignore
                elif annotation.default_dtype:
                    dtype = annotation.default_dtype
                else:
                    dtype = annotation.arg  # type: ignore

            if annotation.origin is None or dtype:
                if check_name is False:
                    raise SchemaInitError(
                        f"'check_name' is not supported for {field_name}."
                    )

                column_kwargs = (
                    field.column_properties(
                        dtype,
                        nullable=annotation.nullable,
                        required=not annotation.is_optional_field,
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
        check_obj: pa.Table,
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

    @classmethod
    def empty(cls: type[Self], *_args) -> Table[Self]:
        """Create an empty pyarrow Table with the schema of this model."""
        schema = cls.to_schema()
        arrow_schema = pa.schema(
            [
                (name, _narwhals_dtype_to_pyarrow(col.dtype))
                for name, col in schema.columns.items()
            ]
        )
        return cast(Table[Self], arrow_schema.empty_table())


def _narwhals_dtype_to_pyarrow(dtype) -> pa.DataType:
    """Translate a narwhals engine dtype back into a ``pyarrow.DataType``.

    Narwhals owns the pyarrow mapping, so build a zero-row frame of the
    narwhals dtype and read the pyarrow type back off it rather than
    maintaining a parallel lookup table here.
    """
    import narwhals.stable.v1 as nw

    empty = nw.from_native(pa.table({"x": pa.array([])}), eager_only=True)
    casted = empty.with_columns(nw.col("x").cast(dtype.type))
    return nw.to_native(casted).schema.field("x").type
