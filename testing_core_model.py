import pandas as pd

from pandera.core.pandas.model import DataFrameModel, SchemaModel
from pandera.core.pandas.model_components import Field
from pandera.typing import Series


class MyModel(DataFrameModel):
    col1: Series[int] = Field(gt=0)
    col2: Series[float] = Field(lt=100)
    col3: Series[str] = Field(isin=[*"ABC"])


class MySchema(SchemaModel):
    col1: Series[int] = Field(gt=0)
    col2: Series[float] = Field(lt=100)
    col3: Series[str] = Field(isin=[*"ABC"])


df = pd.DataFrame(
    {
        "col1": [1, 2, 3],
        "col2": [1.0, 2.0, 3.0],
        "col3": [*"ABC"],
    }
)


print(MyModel.validate(df))
print(MySchema.validate(df))
