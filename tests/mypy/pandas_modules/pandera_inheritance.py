# pylint: skip-file
"""With the pandera.mypy plugin, mypy ignores type overrides."""

import pandera.pandas as pa


class Schema(pa.DataFrameModel):
    a: pa.typing.Series[int]
    b: pa.typing.Series[str]
    c: pa.typing.Series[bool]


class Schema2(Schema):
    a: pa.typing.Series[str]
    b: pa.typing.Series[float]
    c: pa.typing.Series[int]
