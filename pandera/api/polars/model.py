"""Class-based API for Polars models."""

import copy
import inspect
from typing import Optional, Union, cast, overload

import polars as pl
from typing_extensions import Self

from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model import get_dtype_kwargs
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.polars.components import Column
from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.model_config import BaseConfig
from pandera.api.polars.types import PolarsFrame
from pandera.engines import polars_engine as pe
from pandera.errors import SchemaInitError
from pandera.typing import AnnotationInfo
from pandera.typing.polars import DataFrame, LazyFrame, Series
from pandera.utils import docstring_substitution


class DataFrameModel(_DataFrameModel[pl.LazyFrame, DataFrameSchema]):
    """Model of a Polars :class:`~pandera.api.pandas.container.DataFrameSchema`.

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
                is_polars_dtype = inspect.isclass(
                    annotation.raw_annotation
                ) and issubclass(annotation.raw_annotation, pe.DataType)
            except TypeError:
                is_polars_dtype = False

            try:
                engine_dtype = pe.Engine.dtype(annotation.raw_annotation)
                if is_polars_dtype:
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
                    f"Invalid annotation '{field_name}: {annotation.raw_annotation}'."
                )

        return columns

    @classmethod
    @overload
    def validate(
        cls: type[Self],
        check_obj: pl.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> DataFrame[Self]: ...

    @classmethod
    @overload
    def validate(
        cls: type[Self],
        check_obj: pl.LazyFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> LazyFrame[Self]: ...

    @classmethod
    @docstring_substitution(validate_doc=BaseSchema.validate.__doc__)
    def validate(
        cls: type[Self],
        check_obj: PolarsFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Union[LazyFrame[Self], DataFrame[Self]]:
        """%(validate_doc)s"""
        result = cls.to_schema().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )
        if isinstance(check_obj, pl.LazyFrame):
            return cast(LazyFrame[Self], result)
        else:
            return cast(DataFrame[Self], result)

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

        # Define a mapping from Polars data types to JSON schema types
        # This is more robust than string parsing
        POLARS_TO_JSON_TYPE_MAP = {
            # Integer types
            pl.Int8: "integer",
            pl.Int16: "integer",
            pl.Int32: "integer",
            pl.Int64: "integer",
            pl.UInt8: "integer",
            pl.UInt16: "integer",
            pl.UInt32: "integer",
            pl.UInt64: "integer",
            # Float types
            pl.Float32: "number",
            pl.Float64: "number",
            # Boolean type
            pl.Boolean: "boolean",
            # String types
            pl.Utf8: "string",
            pl.String: "string",
            # Date/Time types
            pl.Date: "datetime",
            pl.Datetime: "datetime",
            pl.Time: "datetime",
            pl.Duration: "datetime",
        }

        def map_dtype_to_json_type(dtype):
            """
            Map a Polars data type to a JSON schema type.

            Args:
                dtype: Polars data type

            Returns:
                str: JSON schema type string
            """
            # First try the direct mapping
            if dtype.__class__ in POLARS_TO_JSON_TYPE_MAP:
                return POLARS_TO_JSON_TYPE_MAP[dtype.__class__]

            # Fallback to string representation check for edge cases
            dtype_str = str(dtype).lower()
            if "float" in dtype_str:
                return "number"
            elif "int" in dtype_str:
                return "integer"
            elif "bool" in dtype_str:
                return "boolean"
            elif any(t in dtype_str for t in ["date", "time", "datetime"]):
                return "datetime"
            else:
                return "string"

        properties = {}
        for col_name, col_schema in schema.dtypes.items():
            json_type = map_dtype_to_json_type(col_schema.type)
            properties[col_name] = {
                "type": "array",
                "items": {"type": json_type},
            }

        return {
            "title": schema.name or "pandera.DataFrameSchema",
            "type": "object",
            "properties": properties,
        }

    @classmethod
    def empty(cls: type[Self], *_args) -> DataFrame[Self]:
        """Create an empty DataFrame with the schema of this model."""
        schema = copy.deepcopy(cls.to_schema())
        schema.coerce = True
        empty_df = schema.coerce_dtype(pl.DataFrame(schema=[*schema.columns]))
        return DataFrame[Self](empty_df)
