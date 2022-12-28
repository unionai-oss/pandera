"""Register dask accessor for pandera schema metadata."""

from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from pandera.accessors.pandas_accessor import (
    PanderaDataFrameAccessor,
    PanderaSeriesAccessor,
)

register_dataframe_accessor("pandera")(PanderaDataFrameAccessor)
register_series_accessor("pandera")(PanderaSeriesAccessor)
