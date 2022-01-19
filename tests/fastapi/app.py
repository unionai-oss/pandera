# pylint: skip-file
import pandas as pd
from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse

from pandera.typing import DataFrame
from pandera.typing.fastapi import UploadFile
from tests.fastapi.models import (
    Item,
    ResponseModel,
    Transactions,
    TransactionsDictOut,
    TransactionsJsonToParquet,
)

app = FastAPI()


@app.post("/items/", response_model=Item)
def create_item(item: Item):
    return item


@app.post("/transactions/", response_model=DataFrame[TransactionsDictOut])
def create_transactions(transactions: DataFrame[Transactions]):
    return transactions.assign(name="foo")


@app.post("/file/", response_model=ResponseModel)
def create_upload_file(
    file: UploadFile[DataFrame[TransactionsJsonToParquet]] = File(...),
):
    data = pd.read_parquet(file.data).assign(name="foo")
    return {
        "filename": file.filename,
        "df": data,
    }


@app.get("/")
def main():
    content = """
<body>
<form action="/file/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
