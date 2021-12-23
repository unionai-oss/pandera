# pylint: skip-file
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3]}, index=[*"abc"])
df.loc["a":"c"]
