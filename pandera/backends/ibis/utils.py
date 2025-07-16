"""Utility functions for the Ibis backend."""

from ibis import selectors as s


def select_column(*names):
    """Select a column from a table."""
    if hasattr(s, "cols"):
        return s.cols(*names)
    return s.c(*names)
