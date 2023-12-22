"""Ibis parsing, validation, and error-reporting backends."""

from pandera.backends.ibis.base import IbisSchemaBackend


class DataFrameSchemaBackend(IbisSchemaBackend):
    """Backend for Ibis DataFrameSchema."""
