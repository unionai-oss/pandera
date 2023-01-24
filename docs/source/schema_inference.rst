.. currentmodule:: pandera

.. _schema_inference:

Schema Inference
================

*New in version 0.4.0*

With simple use cases, writing a schema definition manually is pretty
straight-forward with pandera. However, it can get tedious to do this with
dataframes that have many columns of various data types.

To help you handle these cases, the :func:`~pandera.schema_inference.pandas.infer_schema` function enables
you to quickly infer a draft schema from a pandas dataframe or series. Below
is a simple example:

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   import pandas as pd
   import pandera as pa

   from pandera import Check, Column, DataFrameSchema

   df = pd.DataFrame({
       "column1": [5, 10, 20],
       "column2": ["a", "b", "c"],
       "column3": pd.to_datetime(["2010", "2011", "2012"]),
   })
   schema = pa.infer_schema(df)
   print(schema)

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    <Schema DataFrameSchema(
        columns={
            'column1': <Schema Column(name=column1, type=DataType(int64))>
            'column2': <Schema Column(name=column2, type=DataType(object))>
            'column3': <Schema Column(name=column3, type=DataType(datetime64[ns]))>
        },
        checks=[],
        coerce=True,
        dtype=None,
        index=<Schema Index(name=None, type=DataType(int64))>,
        strict=False
        name=None,
        ordered=False,
        unique_column_names=False
    )>


These inferred schemas are **rough drafts** that shouldn't be used for
validation without modification. You can modify the inferred schema to
obtain the schema definition that you're satisfied with.

For :class:`~pandera.core.pandas.container.DataFrameSchema` objects, the following methods create
modified copies of the schema:

* :func:`~pandera.core.pandas.container.DataFrameSchema.add_columns`
* :func:`~pandera.core.pandas.container.DataFrameSchema.remove_columns`
* :func:`~pandera.core.pandas.container.DataFrameSchema.update_column`

For :class:`~pandera.core.pandas.array.SeriesSchema` objects:

* :func:`~pandera.core.pandas.array.SeriesSchema.set_checks`

The section below describes two workflows for persisting and modifying an
inferred schema.

.. _schema persistence:

Schema Persistence
------------------

The schema persistence feature requires a pandera installation with the ``io``
extension. See the :ref:`installation<installation>` instructions for more
details.

There are two ways of persisting schemas, inferred or otherwise.

Write to a Python script
~~~~~~~~~~~~~~~~~~~~~~~~

You can also write your schema to a python script with :func:`~pandera.io.to_script`:

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   # supply a file-like object, Path, or str to write to a file. If not
   # specified, to_script will output the code as a string.
   schema_script = schema.to_script()
   print(schema_script)

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    from pandas import Timestamp
    from pandera import DataFrameSchema, Column, Check, Index, MultiIndex

    schema = DataFrameSchema(
        columns={
            "column1": Column(
                dtype="int64",
                checks=[
                    Check.greater_than_or_equal_to(min_value=5.0),
                    Check.less_than_or_equal_to(max_value=20.0),
                ],
                nullable=False,
                unique=False,
                coerce=False,
                required=True,
                regex=False,
                description=None,
                title=None,
            ),
            "column2": Column(
                dtype="object",
                checks=None,
                nullable=False,
                unique=False,
                coerce=False,
                required=True,
                regex=False,
                description=None,
                title=None,
            ),
            "column3": Column(
                dtype="datetime64[ns]",
                checks=[
                    Check.greater_than_or_equal_to(
                        min_value=Timestamp("2010-01-01 00:00:00")
                    ),
                    Check.less_than_or_equal_to(
                        max_value=Timestamp("2012-01-01 00:00:00")
                    ),
                ],
                nullable=False,
                unique=False,
                coerce=False,
                required=True,
                regex=False,
                description=None,
                title=None,
            ),
        },
        checks=None,
        index=Index(
            dtype="int64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=2.0),
            ],
            nullable=False,
            coerce=False,
            name=None,
            description=None,
            title=None,
        ),
        dtype=None,
        coerce=True,
        strict=False,
        name=None,
        ordered=False,
        unique=None,
        report_duplicates="all",
        unique_column_names=False,
        title=None,
        description=None,
    )

As a python script, you can iterate on an inferred schema and use it to
validate data once you are satisfied with your schema definition.


Write to YAML
~~~~~~~~~~~~~

You can also write the schema object to a yaml file with :func:`~pandera.io.to_yaml`,
and you can then read it into memory with :func:`~pandera.io.from_yaml`. The
:func:`~pandera.core.pandas.container.DataFrameSchema.to_yaml` and :func:`~pandera.core.pandas.container.DataFrameSchema.from_yaml`
is a convenience method for this functionality.

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   # supply a file-like object, Path, or str to write to a file. If not
   # specified, to_yaml will output a yaml string.
   yaml_schema = schema.to_yaml()
   print(yaml_schema.replace(f"{pa.__version__}", "{PANDERA_VERSION}"))

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    schema_type: dataframe
    version: {PANDERA_VERSION}
    columns:
      column1:
        title: null
        description: null
        dtype: int64
        nullable: false
        checks:
          greater_than_or_equal_to: 5.0
          less_than_or_equal_to: 20.0
        unique: false
        coerce: false
        required: true
        regex: false
      column2:
        title: null
        description: null
        dtype: object
        nullable: false
        checks: null
        unique: false
        coerce: false
        required: true
        regex: false
      column3:
        title: null
        description: null
        dtype: datetime64[ns]
        nullable: false
        checks:
          greater_than_or_equal_to: '2010-01-01 00:00:00'
          less_than_or_equal_to: '2012-01-01 00:00:00'
        unique: false
        coerce: false
        required: true
        regex: false
    checks: null
    index:
    - title: null
      description: null
      dtype: int64
      nullable: false
      checks:
        greater_than_or_equal_to: 0.0
        less_than_or_equal_to: 2.0
      name: null
      unique: false
      coerce: false
    dtype: null
    coerce: true
    strict: false
    name: null
    ordered: false
    unique: null
    report_duplicates: all
    unique_column_names: false
    title: null
    description: null

You can edit this yaml file to modify the schema. For example, you can specify
new column names under the ``column`` key, and the respective values map onto
key-word arguments in the :class:`~pandera.core.pandas.components.Column` class.

.. note::

   Currently, only built-in :class:`~pandera.core.checks.Check` methods are supported under the
   ``checks`` key.


Write to JSON
~~~~~~~~~~~~~

Finally, you can also write the schema object to a json file with :func:`~pandera.io.to_json`,
and you can then read it into memory with :func:`~pandera.io.from_json`. The
:func:`~pandera.core.pandas.container.DataFrameSchema.to_json` and :func:`~pandera.core.pandas.container.DataFrameSchema.from_json`
is a convenience method for this functionality.

.. testcode:: infer_dataframe_schema
   :skipif: SKIP

   # supply a file-like object, Path, or str to write to a file. If not
   # specified, to_yaml will output a yaml string.
   json_schema = schema.to_json(indent=4)
   print(json_schema.replace(f"{pa.__version__}", "{PANDERA_VERSION}"))

.. testoutput:: infer_dataframe_schema
   :skipif: SKIP

    {
        "schema_type": "dataframe",
        "version": "{PANDERA_VERSION}",
        "columns": {
            "column1": {
                "title": null,
                "description": null,
                "dtype": "int64",
                "nullable": false,
                "checks": {
                    "greater_than_or_equal_to": 5.0,
                    "less_than_or_equal_to": 20.0
                },
                "unique": false,
                "coerce": false,
                "required": true,
                "regex": false
            },
            "column2": {
                "title": null,
                "description": null,
                "dtype": "object",
                "nullable": false,
                "checks": null,
                "unique": false,
                "coerce": false,
                "required": true,
                "regex": false
            },
            "column3": {
                "title": null,
                "description": null,
                "dtype": "datetime64[ns]",
                "nullable": false,
                "checks": {
                    "greater_than_or_equal_to": "2010-01-01 00:00:00",
                    "less_than_or_equal_to": "2012-01-01 00:00:00"
                },
                "unique": false,
                "coerce": false,
                "required": true,
                "regex": false
            }
        },
        "checks": null,
        "index": [
            {
                "title": null,
                "description": null,
                "dtype": "int64",
                "nullable": false,
                "checks": {
                    "greater_than_or_equal_to": 0.0,
                    "less_than_or_equal_to": 2.0
                },
                "name": null,
                "unique": false,
                "coerce": false
            }
        ],
        "dtype": null,
        "coerce": true,
        "strict": false,
        "name": null,
        "ordered": false,
        "unique": null,
        "report_duplicates": "all",
        "unique_column_names": false,
        "title": null,
        "description": null
    }

You can edit this json file to update the schema as needed, and then load
it back into a pandera schema object with :func:`~pandera.io.from_json` or
:func:`~pandera.core.pandas.container.DataFrameSchema.from_json`.
