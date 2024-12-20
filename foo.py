import pandera as pa
import pandas as pd

schema = pa.DataFrameSchema(
    columns={"a": pa.Column(int, pa.Check.greater_than(0))},
)

schema.validate(pd.DataFrame({"a": [1, 2, 3]}))
