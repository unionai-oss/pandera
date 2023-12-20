"""Core Ibis schema component specifications."""

from pandera.api.pandas.components import Column as _Column


class Column(_Column):
    """Validate types and properties of table columns."""
