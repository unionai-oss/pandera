"""Register koalas accessor for pandera schema metadata."""

from databricks.koalas.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from pandera.pandas_accessor import (
    PanderaDataFrameAccessor,
    PanderaSeriesAccessor,
)

register_dataframe_accessor("pandera")(PanderaDataFrameAccessor)
register_series_accessor("pandera")(PanderaSeriesAccessor)
