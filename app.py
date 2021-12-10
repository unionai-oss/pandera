from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import pandera as pa
from pandera.typing import DataFrame
from pandera.typing.fastapi import UploadFile


class Transactions(pa.SchemaModel):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]

    class Config:
        coerce = True


class TransactionsCsv(Transactions):
    class Config:
        pre_format = "csv"


class TransactionsJson(Transactions):
    class Config:
        pre_format = "json"
        pre_format_options = {"orient": "records"}


class TransactionsJsonToParquet(Transactions):
    class Config:
        pre_format = "json"
        pre_format_options = {"orient": "records"}
        post_format = "parquet"


class TransactionsParquet(Transactions):
    class Config:
        pre_format = "parquet"


class TransactionsFeather(Transactions):
    class Config:
        pre_format = "feather"


class TransactionsPickle(Transactions):
    class Config:
        pre_format = "pickle"


class TransactionsOut(Transactions):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]
    name: pa.typing.Series[str]


class TransactionsJsonOut(TransactionsOut):
    class Config:
        post_format = "json"
        post_format_options = {"orient": "records"}


class TransactionsCsvOut(TransactionsOut):
    class Config:
        post_format = "csv"
        post_format_options = {"index": False}


class Item(BaseModel):
    name: str
    value: int = Field(ge=0)
    description: Optional[str] = None


sample = Transactions.example(size=10)
if not Path("transactions.csv").exists():
    sample.to_csv("transactions.csv", index=False)
if not Path("transactions.json").exists():
    sample.to_json("transactions.json", orient="records")
if not Path("transactions.parquet").exists():
    sample.to_parquet("transactions.parquet")
if not Path("transactions.feather").exists():
    sample.to_feather("transactions.feather")
if not Path("transactions.pickle").exists():
    sample.to_pickle("transactions.pickle")


app = FastAPI()


@app.post("/items/", response_model=Item)
def create_item(item: Item):
    return item


@app.post("/transactions/", response_model=DataFrame[TransactionsJson])
def create_transactions(transactions: DataFrame[Transactions]):
    return transactions


class ResponseModel(BaseModel):
    filename: str
    df: DataFrame[TransactionsJsonOut]


@app.post("/file/", response_model=ResponseModel)
def create_upload_file(
    file: UploadFile[DataFrame[TransactionsJsonToParquet]] = File(...),
):
    data = pd.read_parquet(file.data).assign(name="foo")
    # do stuff with parquet

    data.head(5)
    return {
        "filename": file.filename,
        "df": data.head(5),
    }


@app.get("/")
async def main():
    content = """
<body>
<form action="/file/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
