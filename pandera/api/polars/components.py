"""Schema components for polars."""

from pandera.api.pandas.components import Column as _Column


class Column(_Column):
    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value
