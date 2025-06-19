# pylint: skip-file
from typing import Optional

from pydantic import BaseModel, Field

import pandera.pandas as pa


class Transactions(pa.DataFrameModel):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float] = pa.Field(ge=0, le=1000)

    class Config:
        coerce = True


class TransactionsParquet(Transactions):
    class Config:
        from_format = "parquet"


class TransactionsOut(Transactions):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]
    name: pa.typing.Series[str]


class TransactionsJsonOut(TransactionsOut):
    class Config:
        to_format = "json"
        to_format_kwargs = {"orient": "records"}


class TransactionsDictOut(TransactionsOut):
    class Config:
        to_format = "dict"
        to_format_kwargs = {"orient": "records"}


class Item(BaseModel):
    name: str
    value: int = Field(ge=0)
    description: Optional[str] = None


class ResponseModel(BaseModel):
    filename: str
    df: pa.typing.DataFrame[TransactionsJsonOut]
