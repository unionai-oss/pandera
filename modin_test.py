import os

import modin.pandas as mpd
import numpy as np
import pandas as pd

import pandera as pa

os.environ["MODIN_ENGINE"] = "ray"


if os.environ["MODIN_ENGINE"] == "ray":
    import ray

    ray.init()


schema = pa.DataFrameSchema({"col": pa.Column(int, pa.Check.ge(0))})


mdf = mpd.DataFrame({"col": range(10)})
validated_mdf = schema.validate(mdf)
print(validated_mdf)
print(type(validated_mdf))
