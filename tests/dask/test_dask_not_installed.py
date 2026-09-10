"""Tests behavior when dask is not installed."""

import sys
from unittest import mock

import pandas as pd
import pytest


def test_dask_not_installed() -> None:
    """
    Test that Pandera and its modules can be imported and continue to work
    without dask.
    """
    with mock.patch.dict("sys.modules", {"dask": None}):
        with pytest.raises(ImportError):
            # pylint: disable=import-outside-toplevel,unused-import
            import dask.dataframe

        for module in ["pandera", "pandera.accessors.dask_accessor"]:
            try:
                del sys.modules[module]
            except KeyError:
                ...

        # pylint: disable=import-outside-toplevel,unused-import
        import pandera

        assert "pandera.accessors.dask_accessor" not in sys.modules

        del sys.modules["pandera"]
        del sys.modules["pandera.api.pandas.types"]
        # pylint: disable=import-outside-toplevel
        from pandera.api.pandas.types import is_table

        assert not is_table(pd.Series([1]))

        for module in ["pandera", "pandera.typing"]:
            try:
                del sys.modules[module]
            except KeyError:
                ...

        # pylint: disable=import-outside-toplevel
        import pandera.typing

        annotation = pandera.typing.DataFrame[int]
        assert pandera.typing.AnnotationInfo(annotation).is_generic_df
