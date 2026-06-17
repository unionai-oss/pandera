"""Utility functions for the Ibis backend."""

import ibis.selectors as s


def select_column(*names):
    """Select a column from a table."""
    if hasattr(s, "cols"):
        return s.cols(*names)
    return s.c(*names)
