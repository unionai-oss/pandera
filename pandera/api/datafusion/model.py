"""Class-based API for DataFusion models."""

from __future__ import annotations

import sys
from typing import cast

import datafusion
import pyarrow as pa

from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check
from pandera.api.dataframe.model import DataFrameModel as _DataFrameModel
from pandera.api.dataframe.model_components import FieldInfo
from pandera.api.datafusion.components import Column
from pandera.api.datafusion.container import DataFrameSchema
from pandera.api.datafusion.model_config import BaseConfig
from pandera.api.pyarrow.model import (
    _narwhals_dtype_to_pyarrow,
    build_arrow_columns,
)
from pandera.typing import AnnotationInfo
from pandera.typing.datafusion import DataFrame
from pandera.utils import docstring_substitution

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class DataFrameModel(_DataFrameModel[datafusion.DataFrame, DataFrameSchema]):
    """Definition of a :class:`~pandera.api.datafusion.container.DataFrameSchema`.

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
        # DataFusion dtypes are pyarrow end-to-end, so field annotations
        # resolve with the same rules as the pyarrow model.
        return build_arrow_columns(fields, checks, column_cls=Column)

    @classmethod
    @docstring_substitution(validate_doc=BaseSchema.validate.__doc__)
    def validate(
        cls: type[Self],
        check_obj: datafusion.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> DataFrame[Self]:
        """%(validate_doc)s"""
        result = cls.to_schema().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )
        return cast(DataFrame[Self], result)

    @classmethod
    def empty(cls: type[Self], *_args) -> DataFrame[Self]:
        """Create an empty DataFusion DataFrame with the schema of this model.

        The frame is registered in a fresh
        :class:`datafusion.SessionContext`.
        """
        schema = cls.to_schema()
        arrow_schema = pa.schema(
            [
                (name, _narwhals_dtype_to_pyarrow(col.dtype))
                for name, col in schema.columns.items()
            ]
        )
        ctx = datafusion.SessionContext()
        return cast(
            DataFrame[Self], ctx.from_arrow(arrow_schema.empty_table())
        )
