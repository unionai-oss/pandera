"""Tests behavior when dask is not installed. """
import sys
from unittest import mock

import pandas as pd
import pytest


def test_dask_not_installed() -> None:
    """Test that Pandera and its modules can be imported and continue to work
    without dask"""
    with mock.patch.dict("sys.modules", {"dask": None}):
        with pytest.raises(ImportError):
            # pylint: disable=import-outside-toplevel,unused-import
            import dask.dataframe

        del sys.modules["pandera"]
        del sys.modules["pandera.dask_accessor"]
        # pylint: disable=import-outside-toplevel,unused-import
        import pandera

        assert "pandera.dask_accessor" not in sys.modules

        del sys.modules["pandera"]
        del sys.modules["pandera.check_utils"]
        # pylint: disable=import-outside-toplevel
        import pandera.check_utils

        assert not pandera.check_utils.is_table(pd.Series([1]))

        del sys.modules["pandera"]
        del sys.modules["pandera.typing"]
        # pylint: disable=import-outside-toplevel
        import pandera.typing

        annotation = pandera.typing.DataFrame[int]
        assert pandera.typing.AnnotationInfo(annotation).is_generic_df
