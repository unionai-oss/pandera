"""Schema components for polars."""

from pandera.api.pandas.components import Column as _Column
from pandera.engines import polars_engine


class Column(_Column):
    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = polars_engine.Engine.dtype(value) if value else None
