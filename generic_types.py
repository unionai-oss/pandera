from typing import List, Dict

from pyspark.sql.types import MapType, StringType

import pandera as pa
import pandas as pd


schema = pa.DataFrameSchema(
    {
        "x": pa.Column(Dict[str, str], coerce=True),
        "y": pa.Column(MapType(StringType(), StringType())),
        "z": pa.Column(List[str], coerce=True),
    }
)

data = pd.DataFrame(
    {
        "x": [{1: "bar"}, {2: "foo"}],
        "y": [{1: "bar"}, {2: "foo"}],
        "z": [[1, 2], [3]],
    }
)

validated = schema(data)
print(validated)
print(validated["z"].map(lambda x: [type(i) for i in x]))
