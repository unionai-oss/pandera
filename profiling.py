from pyinstrument import Profiler

profiler = Profiler()

import pandera as pa
import pandas as pd


profiler.start()
schema = pa.DataFrameSchema(
    {
        "column1": pa.Column(int),
        "column2": pa.Column(float),
        "column3": pa.Column(str),
    }
)

print(schema)

schema.validate(
    pd.DataFrame(
        {
            "column1": [1, 2, 3],
            "column2": [4, 5, 6],
            "column3": ["a", "b", "c"],
        }
    )
)

# profiler.stop()
# profiler.print()
# print(profiler.output_html())
