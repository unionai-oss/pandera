import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import pydantic


class SimpleSchema(pa.SchemaModel):
    str_col: Series[str] = pa.Field(unique=True, isin=list("abcd"))
    int_col: Series[int]
    float_col: Series[float]


class PydanticModel(pydantic.BaseModel):
    x: int
    df: DataFrame[SimpleSchema]


@pydantic.validate_arguments
def fn(x: int, df: DataFrame[SimpleSchema]):
    return x, df


print(fn(1, pd.DataFrame({"str_col": ["a"], "int_col": [1], "float_col": [1.0]})))


print(PydanticModel.schema_json(indent=4))
