```{eval-rst}
.. currentmodule:: pandera
```

(fastapi-integration)=

# FastAPI

*new in 0.9.0*

Since both FastAPI and Pandera integrates seamlessly with Pydantic, you can
use the {py:class}`~pandera.api.pandas.model.DataFrameModel` types to validate incoming
or outgoing data with respect to your API endpoints.

## Using DataFrameModels to Validate Endpoint Inputs and Outputs

Suppose we want to process transactions, where each transaction has an
`id` and `cost`. We can model this with a pandera dataframe model:

```{literalinclude} ../../tests/fastapi/models.py
:language: python
:lines: 1-14
```

Also suppose that we expect our endpoint to add a `name` to the transaction
data:

```{literalinclude} ../../tests/fastapi/models.py
:language: python
:lines: 22-25
```

Let's also assume that the output of the endpoint should be a list of dictionary
records containing the named transactions data. We can do this easily with the
`to_format` option in the dataframe model {py:class}`~pandera.typing.config.BaseConfig`.

```{literalinclude} ../../tests/fastapi/models.py
:language: python
:lines: 34-37
```

Note that the `to_format_kwargs` is a dictionary of key-word arguments
to be passed into the respective pandas `to_{format}` method.

% TODO: create new page for the to/from_format config option

Next we'll create a FastAPI app and define a `/transactions/` POST endpoint:

```{literalinclude} ../../tests/fastapi/app.py
:language: python
:lines: 2-6,14-21,28-34
```

## Reading File Uploads

Similar to the `TransactionsDictOut` example to convert dataframes to a
particular format as an endpoint response, pandera also provides a
`from_format` dataframe model configuration option to read a dataframe from
a particular serialization format.

```{literalinclude} ../../tests/fastapi/models.py
:language: python
:lines: 17-19
```

Let's also define a response model for the `/file/` upload endpoint:

```{literalinclude} ../../tests/fastapi/models.py
:language: python
:lines: 28-32,46-48
```

In the next example, we use the pandera
{py:class}`~pandera.typing.fastapi.UploadFile` type to upload a parquet file
to the `/file/` POST endpoint and return a response containing the filename
and the modified data in json format.

```{literalinclude} ../../tests/fastapi/app.py
:language: python
:lines: 37-44
```

Pandera's {py:class}`~pandera.typing.fastapi.UploadFile` type is a subclass of FastAPI's
[UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/?h=uploadfile#uploadfile)
but it exposes a `.data` property containing the pandera-validated dataframe.

## Takeaway

With the FastAPI and Pandera integration, you can use Pandera
{py:class}`~pandera.api.pandas.model.DataFrameModel` types to validate the dataframe inputs
and outputs of your FastAPI endpoints.
