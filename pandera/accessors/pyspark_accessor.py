# pylint: skip-file
# NOTE: skip file since py=3.10 yields these errors:
# https://github.com/pandera-dev/pandera/runs/4998710717?check_suite_focus=true
"""Register pyspark accessor for pandera schema metadata."""

from pyspark.pandas.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from pandera.accessors.pandas_accessor import (
    PanderaDataFrameAccessor,
    PanderaSeriesAccessor,
)

register_dataframe_accessor("pandera")(PanderaDataFrameAccessor)
register_series_accessor("pandera")(PanderaSeriesAccessor)
