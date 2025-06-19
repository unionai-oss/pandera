# pylint: skip-file
from fastapi import Body, FastAPI, File
from fastapi.responses import HTMLResponse

from pandera.typing import DataFrame
from pandera.typing.fastapi import UploadFile
from tests.fastapi.models import (
    Item,
    ResponseModel,
    Transactions,
    TransactionsDictOut,
    TransactionsParquet,
)

try:
    from typing import Annotated  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Annotated  # type: ignore[assignment]

app = FastAPI()


@app.post("/items/", response_model=Item)
def create_item(item: Item):
    return item


@app.post("/transactions/", response_model=DataFrame[TransactionsDictOut])
def create_transactions(
    transactions: Annotated[DataFrame[Transactions], Body()],
):
    output = transactions.assign(name="foo")
    ...  # do other stuff, e.g. update backend database with transactions
    return output


@app.post("/file/", response_model=ResponseModel)
def create_upload_file(
    file: Annotated[UploadFile[DataFrame[TransactionsParquet]], File()],
):
    return {
        "filename": file.filename,
        "df": file.data.assign(name="foo"),
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
