"""DataFrame Schema for Polars."""

from typing import Optional

import polars as pl

from pandera.api.pandas.container import DataFrameSchema as _DataFrameSchema
from pandera.dtypes import DataType
from pandera.engines import polars_engine


class DataFrameSchema(_DataFrameSchema):
    def validate(
        self,
        check_obj: pl.LazyFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pl.LazyFrame:
        """Validate a polars DataFrame against the schema."""

        return self.get_backend(check_obj).validate(
            check_obj=check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value) -> None:
        """Set the pandas dtype property."""
        self._dtype = polars_engine.Engine.dtype(value) if value else None
