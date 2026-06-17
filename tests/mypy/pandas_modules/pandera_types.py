# pylint: skip-file
import pandas as pd

import pandera.pandas as pa


def fn(series: pa.typing.Series[int]) -> None:
    pass


df = pd.DataFrame({"a": [1, 2, 3]})
sr = pd.Series([1, 2, 3])

fn(sr)
fn(df["a"])
