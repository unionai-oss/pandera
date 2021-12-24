from io import BytesIO, StringIO

import pandas as pd
import pydantic
from pydantic.decorator import validate_arguments

import pandera as pa
from pandera.typing import DataFrame, Series


class SimpleSchema(pa.SchemaModel):
    str_col: Series[str] = pa.Field(unique=True, isin=list("abcd"))
    int_col: Series[int]
    float_col: Series[float]

    class Config:
        from_format = "parquet"


class OutSchema(SimpleSchema):
    class Config:
        from_format = None
        to_format = "feather"


class PydanticModel(pydantic.BaseModel):
    x: int
    df: DataFrame[OutSchema]


# @validate_arguments
@pa.check_types
def fn(x: int, df: DataFrame[SimpleSchema]) -> DataFrame[OutSchema]:
    return df.assign(foo=x)
    # return PydanticModel(x=x, df=df.assign(foo=x))


df = pd.DataFrame({"str_col": ["a"], "int_col": [1], "float_col": [1.0]})
buf = BytesIO()
df.to_parquet(buf, index=False)
buf.seek(0)
print(fn(1, buf))


# print(PydanticModel.schema_json(indent=4))
