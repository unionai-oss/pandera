.. currentmodule:: pandera

.. _data-format-conversion:

Data Format Conversion ⭐️ (New)
=================================

*new in 0.9.0*

The class-based API provides configuration options for converting data to/from
supported serialization formats in the context of
:py:func:`~pandera.decorators.check_types` -decorated functions.

.. note::

   Currently, :py:class:`pandera.typing.pandas.DataFrame` is the only data
   type that supports this feature.

Consider this simple example:

.. testcode:: format_serialization

    import pandera as pa
    from pandera.typing import DataFrame, Series

    class InSchema(pa.SchemaModel):
        str_col: Series[str] = pa.Field(unique=True, isin=[*"abcd"])
        int_col: Series[int]

    class OutSchema(InSchema):
        float_col: pa.typing.Series[float]

    @pa.check_types
    def transform(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
        return df.assign(float_col=1.1)


With the schema type annotations and
:py:func:`~pandera.decorators.check_types` decorator, the ``transform``
function validates DataFrame inputs and outputs according to the ``InSchema``
and ``OutSchema`` definitions.

But what if your input data is serialized in parquet format, and you want to
read it into memory, validate the DataFrame, and then pass it to a downstream
function for further analysis? Similarly, what if you want the output of
``transform`` to be a list of dictionary records instead of a pandas DataFrame?

The ``to/from_format`` Configuration Options
--------------------------------------------

To easily fulfill the use cases described above, you can implement the
read/write logic by hand, or you can configure schemas to do so. We can first
define a subclass of ``InSchema`` with additional configuration so that our
``transform`` function can read data directly from parquet files or buffers:

.. testcode:: format_serialization

    class InSchemaParquet(InSchema):
        class Config:
            from_format = "parquet"

Then, we define subclass of ``OutSchema`` to specify that ``transform``
should output a list of dictionaries representing the rows of the output
dataframe.

.. testcode:: format_serialization

    class OutSchemaDict(OutSchema):
        class Config:
            to_format = "dict"
            to_format_kwargs = {"orient": "records"}

Note that the ``{to/from}_format_kwargs`` configuration option should be
supplied with a dictionary of key-word arguments to be passed into the
respective pandas ``to_{format}`` method.

Finally, we redefine our ``transform`` function:

.. testcode:: format_serialization

    @pa.check_types
    def transform(df: DataFrame[InSchemaParquet]) -> DataFrame[OutSchemaDict]:
        return df.assign(float_col=1.1)


We can test this out using a buffer to store the parquet file.

.. note::
    A string or path-like object representing the filepath to a parquet file
    would also be a valid input to ``transform``.

.. testcode:: format_serialization

    import io
    import json

    buffer = io.BytesIO()
    data = pd.DataFrame({"str_col": [*"abc"], "int_col": range(3)})
    data.to_parquet(buffer)
    buffer.seek(0)

    dict_output = transform(buffer)
    print(json.dumps(dict_output, indent=4))

.. testoutput:: format_serialization

    [
        {
            "str_col": "a",
            "int_col": 0,
            "float_col": 1.1
        },
        {
            "str_col": "b",
            "int_col": 1,
            "float_col": 1.1
        },
        {
            "str_col": "c",
            "int_col": 2,
            "float_col": 1.1
        }
    ]


Takeaway
--------

Data Format Conversion using the ``{to/from}_format`` configuration option
can modify the behavior of :py:func:`~pandera.decorators.check_types` -decorated
functions to convert input data from a particular serialization format into
a dataframe. Additionally, you can convert the output data from a dataframe to
potentially another format.

This dovetails well with the :ref:`FastAPI Integration <fastapi_integration>`
for validating the inputs and outputs of app endpoints.
