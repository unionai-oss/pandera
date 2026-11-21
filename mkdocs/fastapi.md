# FastAPI

*new in 0.9.0*

Since both FastAPI and Pandera integrates seamlessly with Pydantic, you can
use the [`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) types to validate incoming
or outgoing data with respect to your API endpoints.

## Using DataFrameModels to Validate Endpoint Inputs and Outputs

Suppose we want to process transactions, where each transaction has an
`id` and `cost`. We can model this with a pandera dataframe model:

```python
import pandera.pandas as pa


class Transactions(pa.DataFrameModel):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float] = pa.Field(ge=0, le=1000)

    class Config:
        coerce = True
```

Also suppose that we expect our endpoint to add a `name` to the transaction
data:

```python
class TransactionsOut(Transactions):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]
    name: pa.typing.Series[str]
```

Let's also assume that the output of the endpoint should be a list of dictionary
records containing the named transactions data. We can do this easily with the
`to_format` option in the dataframe model [`BaseConfig`](./reference/generated/pandera.api.pandas.model_config.BaseConfig.html).

```python
class TransactionsDictOut(TransactionsOut):
    class Config:
        to_format = "dict"
        to_format_kwargs = {"orient": "records"}
```

Note that the `to_format_kwargs` is a dictionary of key-word arguments
to be passed into the respective pandas `to_{format}` method.

Next we'll create a FastAPI app and define a `/transactions/` POST endpoint:

```python
from typing import Annotated

from fastapi import Body, FastAPI, File
from fastapi.responses import HTMLResponse

from pandera.typing import DataFrame
from pandera.typing.fastapi import UploadFile

app = FastAPI()


@app.post("/transactions/", response_model=DataFrame[TransactionsDictOut])
def create_transactions(
    transactions: Annotated[DataFrame[Transactions], Body()],
):
    output = transactions.assign(name="foo")
    ...  # do other stuff, e.g. update backend database with transactions
    return output
```

## Reading File Uploads

Similar to the `TransactionsDictOut` example to convert dataframes to a
particular format as an endpoint response, pandera also provides a
`from_format` dataframe model configuration option to read a dataframe from
a particular serialization format.

```python
class TransactionsParquet(Transactions):
    class Config:
        from_format = "parquet"
```

Let's also define a response model for the `/file/` upload endpoint:

```python
from pydantic import BaseModel


class TransactionsJsonOut(TransactionsOut):
    class Config:
        to_format = "json"
        to_format_kwargs = {"orient": "records"}


class ResponseModel(BaseModel):
    filename: str
    df: pa.typing.DataFrame[TransactionsJsonOut]
```

In the next example, we use the pandera
[`UploadFile`](./reference/generated/pandera.typing.fastapi.UploadFile.html) type to upload a parquet file
to the `/file/` POST endpoint and return a response containing the filename
and the modified data in json format.

```python
@app.post("/file/", response_model=ResponseModel)
def create_upload_file(
    file: Annotated[UploadFile[DataFrame[TransactionsParquet]], File()],
):
    return {
        "filename": file.filename,
        "df": file.data.assign(name="foo"),
    }
```

Pandera's [`UploadFile`](./reference/generated/pandera.typing.fastapi.UploadFile.html) type is a subclass of FastAPI's
[UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/?h=uploadfile#uploadfile)
but it exposes a `.data` property containing the pandera-validated dataframe.

## Takeaway

With the FastAPI and Pandera integration, you can use Pandera
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) types to validate the dataframe inputs
and outputs of your FastAPI endpoints.
