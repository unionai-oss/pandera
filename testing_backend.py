import pandas as pd
import pandera as pa
from pandera.backends.pandas import (
    PandasSchemaContainerBackend, PandasSchemaFieldBackend
)


col_schema = pa.Column(int, pa.Check.gt(0), name="col")
df_schema = pa.DataFrameSchema({"col": col_schema})

df = pd.DataFrame({"col": [1, 2, 3]})

field_backend = PandasSchemaFieldBackend()
# validated_df = field_backend.validate(df, col_schema)

container_backend = PandasSchemaContainerBackend()
validated_df = container_backend.validate(df, df_schema)
print(validated_df)
