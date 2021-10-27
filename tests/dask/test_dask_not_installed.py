"""Tests behavior when dask is not installed. """

import importlib
from unittest import mock

import pandas as pd
import pytest

import pandera.check_utils
import pandera.typing


def test_dask_not_installed() -> None:
    """Test that Pandera and its modules can be imported and continue to work
    without dask"""
    with mock.patch.dict("sys.modules", {"dask": None}):
        with pytest.raises(ImportError):
            # pylint: disable=reimported,import-outside-toplevel,unused-import
            import dask.dataframe

        importlib.reload(pandera.check_utils)
        assert not pandera.check_utils.is_table(pd.Series([1]))

        importlib.reload(pandera.typing)
        annotation = pandera.typing.DataFrame[int]
        assert pandera.typing.AnnotationInfo(annotation).is_generic_df
