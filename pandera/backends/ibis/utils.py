"""Utility functions for the Ibis backend."""

import ibis
import ibis.selectors as s
from ibis import _


def select_column(*names):
    """Select a column from a table."""
    if hasattr(s, "cols"):
        return s.cols(*names)
    return s.c(*names)


def duplicated_rows(keep, subset):
    """Deferred expression flagging the duplicated rows of a table.

    :param keep: ``"first"``, ``"last"`` or ``False``, as returned by
        :func:`pandera.backends.utils.convert_uniquesettings`.
    :param subset: names of the columns that define a duplicate.
    """
    if keep == "first":
        return ibis.row_number().over(group_by=subset) > 0
    if keep == "last":
        return (_.count() - ibis.row_number()).over(group_by=subset) > 1
    return _.count().over(group_by=subset) > 1
