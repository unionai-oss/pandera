import pandas as pd
import pandera.pandas as pa


df = pd.DataFrame({"col1": ["a", "b"]}, dtype="string[pyarrow]")
schema = pa.DataFrameSchema(columns={"col1": pa.Column("string[pyarrow]")})
schema.validate(df)
