from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import pandera as pa


class Transactions(pa.SchemaModel):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]

    class Config:
        coerce = True


class Item(BaseModel):
    name: str
    description: Optional[str] = None


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/items/")
def create_item(item: Item):
    return item


@app.post("/transactions/", response_model=pa.typing.IndexedColumns[Transactions])
def create_transactions(transactions: pa.typing.DataFrame[Transactions]):
    return transactions
