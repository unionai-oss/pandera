"""GeoPandas :class:`geopandas.GeoDataFrame` model API."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import pandas as pd

from pandera.api.base.schema import BaseSchema
from pandera.api.geopandas.common import to_geodataframe
from pandera.api.pandas.model import DataFrameModel
from pandera.import_utils import strategy_import_error
from pandera.utils import docstring_substitution

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if TYPE_CHECKING:
    from pandera.typing.geopandas import GeoDataFrame as GeoDataFrameT


class GeoDataFrameModel(DataFrameModel):
    """Class-based schema for :class:`geopandas.GeoDataFrame` data.

    Inherits :class:`~pandera.api.pandas.model.DataFrameModel` and reuses the
    same schema-building and validation logic. Use this model when
    :meth:`validate`, :meth:`example`, :meth:`empty`, and :meth:`strategy` should
    return a :class:`geopandas.GeoDataFrame` (preserving geometry columns and CRS
    metadata) even if the pandas backend produced a plain
    :class:`pandas.DataFrame`.

    Requires the ``geopandas`` extra. Use with
    :class:`pandera.typing.geopandas.GeoDataFrame` for validate-on-init, e.g.
    ``GeoDataFrame[MyModel](...)``.
    """

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
    ) -> GeoDataFrameT[Self]:
        """%(validate_doc)s"""
        result = cls.to_schema().validate(
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )
        return cast(
            "GeoDataFrameT[Self]",
            to_geodataframe(result),
        )

    @classmethod
    @docstring_substitution(example_doc=BaseSchema.example.__doc__)
    @strategy_import_error
    def example(cls: type[Self], **kwargs) -> GeoDataFrameT[Self]:
        """%(example_doc)s"""
        out = cls.__schema__.example(**kwargs)
        return cast(
            "GeoDataFrameT[Self]",
            to_geodataframe(out),
        )

    @classmethod
    @strategy_import_error
    def strategy(cls: type[Self], **kwargs):
        strat = cls.__schema__.strategy(**kwargs)
        return strat.map(to_geodataframe)

    @classmethod
    def empty(cls: type[Self], *_args) -> GeoDataFrameT[Self]:
        """Create an empty :class:`geopandas.GeoDataFrame` with this schema."""
        pdf = DataFrameModel.empty.__func__(cls, *_args)  # type: ignore [attr-defined]
        return cast(
            "GeoDataFrameT[Self]",
            to_geodataframe(pdf),
        )
