"""New example."""

import pandas as pd
import pandera.pandas as pa

# data to validate
df = pd.DataFrame(
    {
        "column1": [1, 2, 3],
        "column2": [1.1, 1.2, 1.3],
        "column3": ["a", "b", "c"],
    }
)

schema = pa.DataFrameSchema(
    {
        "column1": pa.Column(int, pa.Check.ge(0)),
        "column2": pa.Column(float, pa.Check.lt(10)),
        "column3": pa.Column(
            str,
            [
                pa.Check.isin([*"abc"]),
                pa.Check(lambda series: series.str.len() == 1),
            ],
        ),
    }
)

print(schema.validate(df))


# define DataFrameModel Schema
class Schema(pa.DataFrameModel):
    column1: int = pa.Field(ge=0)
    column2: float = pa.Field(lt=10)
    column3: str = pa.Field(isin=[*"abc"])

    @pa.check("column3")
    @classmethod
    def custom_check(cls, series: pd.Series) -> pd.Series:
        return series.str.len() == 1


print(Schema.validate(df))
