"""Module for unit testing validation on initialization."""

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame


class ExampleSchema(pa.DataFrameModel):
    class Config:
        coerce = True

    a: int


ExampleDataFrame = DataFrame[ExampleSchema]
validated_dataframe = ExampleDataFrame(pd.DataFrame([], columns=["a"]))
