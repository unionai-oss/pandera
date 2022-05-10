import pandas as pd

import pandera as pa
from pandera.core.pandas.array import ArraySchema

schema = ArraySchema(int, pa.Check.gt(0), name="foo")

data = pd.Series([1, 2, 3, -1], name="foo")
print(schema(data))
