"""Tests for the _unwrap_failure_cases helper in pandera.api.narwhals.utils."""

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from pandera.api.narwhals.utils import _unwrap_failure_cases


class TestUnwrapFailureCases:
    """Tests for _unwrap_failure_cases behavior per the spec."""

    def test_none_passthrough(self):
        """Non-narwhals value None passes through unchanged."""
        assert _unwrap_failure_cases(None) is None

    def test_non_narwhals_passthrough(self):
        """Raw native frames (not narwhals-wrapped) pass through unchanged."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _unwrap_failure_cases(df)
        assert result is df

    def test_polars_lazy_frame_collects_and_unwraps(self):
        """nw.LazyFrame (Polars) is collected and unwrapped to pl.DataFrame."""
        pl_df = pl.DataFrame({"a": [1, 2, 3]})
        nw_lf = nw.from_native(pl_df.lazy())
        assert isinstance(nw_lf, nw.LazyFrame)
        result = _unwrap_failure_cases(nw_lf)
        assert isinstance(result, pl.DataFrame)
        assert result.to_dict(as_series=False) == {"a": [1, 2, 3]}

    def test_polars_eager_dataframe_unwraps(self):
        """nw.DataFrame wrapping a polars eager frame is unwrapped to pl.DataFrame."""
        pl_df = pl.DataFrame({"a": [1, 2, 3]})
        nw_df = nw.from_native(pl_df)
        assert isinstance(nw_df, nw.DataFrame)
        result = _unwrap_failure_cases(nw_df)
        assert isinstance(result, pl.DataFrame)
        assert result.to_dict(as_series=False) == {"a": [1, 2, 3]}

    def test_polars_empty_dataframe_unwraps(self):
        """Empty narwhals DataFrame also unwraps to native without error."""
        pl_df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
        nw_df = nw.from_native(pl_df)
        result = _unwrap_failure_cases(nw_df)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_integer_passthrough(self):
        """Plain scalar values pass through unchanged."""
        assert _unwrap_failure_cases(42) == 42

    def test_string_passthrough(self):
        """Plain string values pass through unchanged."""
        assert _unwrap_failure_cases("failure") == "failure"

    @pytest.mark.ibis
    def test_ibis_table_passthrough(self):
        """nw.DataFrame wrapping an ibis table (SQL-lazy) is unwrapped to ibis.Table."""
        import ibis
        import pandas as pd

        t = ibis.memtable(pd.DataFrame({"a": [1, 2, 3]}))
        nw_df = nw.from_native(t, eager_or_interchange_only=False)
        assert isinstance(nw_df, nw.DataFrame)
        result = _unwrap_failure_cases(nw_df)
        # SQL-lazy: returned as native without executing the query
        assert type(result).__module__.startswith("ibis"), (
            f"Expected ibis type, got {type(result)}"
        )
        assert result is not nw_df  # confirms it was unwrapped
