# Airspeed Velocity Benchmarks for pandera
import pandas as pd

from pandera.pandas import (
    Column,
    DataFrameSchema,
    SeriesSchema,
    Bool,
    Category,
    Check,
    DateTime,
    Float,
    Int,
    Object,
    String,
    Timedelta,
    String,
)


class Validate:
    """
    Benchmarking Series schema.validate
    """

    def setup(self):
        self.schema = SeriesSchema(
            String,
            checks=[
                Check(lambda s: s.str.startswith("foo")),
                Check(lambda s: s.str.endswith("bar")),
                Check(lambda x: len(x) > 3, element_wise=True),
            ],
            nullable=False,
            unique=False,
            name="my_series",
        )
        self.series = pd.Series(["foobar", "foobar", "foobar"], name="my_series")

    def time_series_schema(self):
        self.schema.validate(self.series)

    def mem_series_schema(self):
        self.schema.validate(self.series)

    def peakmem_series_schema(self):
        self.schema.validate(self.series)
