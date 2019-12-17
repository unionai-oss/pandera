import pandas as pd
from docs.source import conf

def test_sphinx_doctest_setting_global_pandas_conditions():
    exec(conf.doctest_global_setup)

    max_cols_after_being_set = pd.options.display.max_columns
    max_rows_after_being_set = pd.options.display.max_rows
    assert max_cols_after_being_set == None
    assert max_rows_after_being_set == None