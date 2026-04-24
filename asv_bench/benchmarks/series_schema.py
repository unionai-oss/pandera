# Airspeed Velocity Benchmarks for pandera
import pandas as pd

from pandera.pandas import (Bool, Category, Check, Column, DataFrameSchema,
                            DateTime, Float, Int, Object, SeriesSchema, String,
                            Timedelta)


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
