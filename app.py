from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import pandera as pa
from pandera.typing import DataFrame
from pandera.typing.fastapi import UploadFile as PanderaUploadFile


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
        post_format = "dict"
        post_format_options = {"orient": "records"}


class Item(BaseModel):
    name: str
    value: int = Field(ge=0)
    description: Optional[str] = None


app = FastAPI()


@app.post("/items/", response_model=Item)
def create_item(item: Item):
    return item


@app.post("/transactions/", response_model=DataFrame[TransactionsJson])
def create_transactions(transactions: DataFrame[Transactions]):
    return transactions


class ResponseModel(BaseModel):
    filename: str
    df: DataFrame[TransactionsJson]


@app.post("/file/", response_model=ResponseModel)
async def create_upload_file(
    file: PanderaUploadFile[DataFrame[TransactionsCsv]] = File(...),
):
    return {"filename": file.filename, "df": pd.read_csv(file.file)}


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
