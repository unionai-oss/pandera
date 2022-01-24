# pylint: skip-file
from typing import Optional

from pydantic import BaseModel, Field

import pandera as pa


class Transactions(pa.SchemaModel):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]

    class Config:
        coerce = True


class TransactionsCsv(Transactions):
    class Config:
        from_format = "csv"


class TransactionsJson(Transactions):
    class Config:
        from_format = "json"
        from_format_options = {"orient": "records"}


class TransactionsJsonToParquet(Transactions):
    class Config:
        from_format = "json"
        from_format_options = {"orient": "records"}
        to_format = "parquet"


class TransactionsParquet(Transactions):
    class Config:
        from_format = "parquet"


class TransactionsFeather(Transactions):
    class Config:
        from_format = "feather"


class TransactionsPickle(Transactions):
    class Config:
        from_format = "pickle"


class TransactionsOut(Transactions):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]
    name: pa.typing.Series[str]


class TransactionsJsonOut(TransactionsOut):
    class Config:
        to_format = "json"
        to_format_options = {"orient": "records"}


class TransactionsDictOut(TransactionsOut):
    class Config:
        to_format = "dict"
        to_format_options = {"orient": "records"}


class TransactionsCsvOut(TransactionsOut):
    class Config:
        to_format = "csv"
        to_format_options = {"index": False}


class Item(BaseModel):
    name: str
    value: int = Field(ge=0)
    description: Optional[str] = None


class ResponseModel(BaseModel):
    filename: str
    df: pa.typing.DataFrame[TransactionsJsonOut]
