"""Module for unit testing validation on initialization."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame


class ExampleSchema(pa.DataFrameModel):
    class Config:
        coerce = True

    a: np.int64


ExampleDataFrame = DataFrame[ExampleSchema]
validated_dataframe = ExampleDataFrame(pd.DataFrame([], columns=["a"]))
