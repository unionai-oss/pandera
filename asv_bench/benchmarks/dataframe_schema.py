# Airspeed Velocity Benchmarks for pandera
import pandas as pd

from pandera.pandas import (
    Column,
    DataFrameSchema,
    Bool,
    Category,
    Check,
    DateTime,
    Float,
    Int,
    Object,
    String,
    Timedelta,
    check_input,
    check_output,
)


class Validate:
    """
    Benchmarking schema.validate
    """

    def setup(self):
        self.schema = DataFrameSchema(
            {
                "a": Column(Int),
                "b": Column(Float),
                "c": Column(String),
                "d": Column(Bool),
                "e": Column(Category),
                "f": Column(Object),
                "g": Column(DateTime),
                "i": Column(Timedelta),
            },
        )
        self.df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.1, 2.5, 9.9],
                "c": ["z", "y", "x"],
                "d": [True, True, False],
                "e": pd.Series(["c2", "c1", "c3"], dtype="category"),
                "f": [(3,), (2,), (1,)],
                "g": [
                    pd.Timestamp("2015-02-01"),
                    pd.Timestamp("2015-02-02"),
                    pd.Timestamp("2015-02-03"),
                ],
                "i": [
                    pd.Timedelta(1, unit="D"),
                    pd.Timedelta(5, unit="D"),
                    pd.Timedelta(9, unit="D"),
                ],
            }
        )

    def time_df_schema(self):
        self.schema.validate(self.df)

    def mem_df_schema(self):
        self.schema.validate(self.df)

    def peakmem_df_schema(self):
        self.schema.validate(self.df)


class Decorators:
    """
    Benchmarking input and output decorator performance.
    """

    def transformer(df):
        return df.assign(column2=[1, 2, 3])

    def setup(self):
        self.in_schema = DataFrameSchema({"column1": Column(String)})
        self.out_schema = DataFrameSchema({"column2": Column(Int)})
        self.df = pd.DataFrame({"column1": ["a", "b", "c"]})

    def time_check_input(self):
        @check_input(self.in_schema)
        def transform_first_arg(self):
            return Decorators.transformer(self.df)

    def mem_check_input(self):
        @check_input(self.in_schema)
        def transform_first_arg(self):
            return Decorators.transformer(self.df)

    def peakmem_check_input(self):
        @check_input(self.in_schema)
        def transform_first_arg(self):
            return Decorators.transformer(self.df)

    def time_check_output(self):
        @check_output(self.out_schema)
        def transform_first_arg(self):
            return Decorators.transformer(self.df)

    def mem_check_output(self):
        @check_output(self.out_schema)
        def transform_first_arg(self):
            return Decorators.transformer(self.df)

    def peakmem_check_output(self):
        @check_output(self.out_schema)
        def transform_first_arg(self):
            return Decorators.transformer(self.df)
