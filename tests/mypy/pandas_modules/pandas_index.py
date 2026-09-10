# pylint: skip-file
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3]})
sr = pd.Series([1, 2, 3])
idx = pd.Index([1, 2, 3])

df_index_unique: bool = df.index.is_unique
sr_index_unique: bool = df["a"].index.is_unique
idx_unique: bool = idx.is_unique
