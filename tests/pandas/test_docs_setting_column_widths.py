"""Some of the doctest examples only work if the terminal is the correct width
because of the way __str__/__repr__ works in pandas. This checks that
conditions necessary for the doctests to pass properly exist on the host
system."""

import pandas as pd

from docs.source import conf


def test_sphinx_doctest_setting_global_pandas_conditions() -> None:
    """Checks that no limit is set on the height/width of the __repr__/__str__
    print of a pd.DataFrame to ensure doctest performs consistently across
    different Operating Systems."""
    # pylint: disable=W0122
    exec(conf.doctest_global_setup)

    max_cols_after_being_set = pd.options.display.max_columns
    max_rows_after_being_set = pd.options.display.max_rows
    assert max_cols_after_being_set is None
    assert max_rows_after_being_set is None
