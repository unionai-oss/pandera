"""GeoPandas :class:`geopandas.GeoDataFrame` schema API."""

from __future__ import annotations

import warnings

import pandas as pd

from pandera.api.geopandas.common import require_geopandas, to_geodataframe
from pandera.api.pandas.container import DataFrameSchema
from pandera.config import get_config_context
from pandera.import_utils import strategy_import_error


class GeoDataFrameSchema(DataFrameSchema):
    """Subclass of :class:`~pandera.api.pandas.container.DataFrameSchema` that
    returns a :class:`geopandas.GeoDataFrame` from :meth:`validate`,
    :meth:`example`, and :meth:`strategy`.

    Use the same constructor arguments as :class:`DataFrameSchema` (``columns``,
    ``checks``, ``index``, etc.). Validation uses the pandas backend; the result
    is coerced to a ``GeoDataFrame`` so geometry columns and CRS metadata are
    preserved for geospatial workflows.

    Requires the ``geopandas`` extra.
    """

    def validate(
        self,
        check_obj: pd.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Like :meth:`DataFrameSchema.validate`, but returns a ``GeoDataFrame``
        when the result is a plain :class:`pandas.DataFrame`.

        If validation returns a Dask object (e.g. Dask DataFrame), it is passed
        through unchanged so distributed workflows are not broken.
        """
        if not get_config_context().validation_enabled:
            return check_obj

        result = super().validate(
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )
        if hasattr(result, "dask"):
            return result
        return to_geodataframe(result)

    @strategy_import_error
    def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
        """Like :meth:`DataFrameSchema.strategy`, but generated frames are
        ``GeoDataFrame`` instances when applicable.
        """
        require_geopandas()
        strat = super().strategy(size=size, n_regex_columns=n_regex_columns)
        return strat.map(to_geodataframe)

    def example(
        self, size: int | None = None, n_regex_columns: int = 1
    ) -> pd.DataFrame:
        """Like :meth:`DataFrameSchema.example`, but returns a ``GeoDataFrame``
        when applicable.
        """
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(
                size=size, n_regex_columns=n_regex_columns
            ).example()
