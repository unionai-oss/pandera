---
file_format: mystnb
---

```{currentmodule} pandera.pyspark
```

(native-pyspark)=

# Data Validation with Pyspark SQL

*new in 0.16.0*

Apache Spark is an open-source unified analytics engine for large-scale data
processing. Spark provides an interface for programming clusters with implicit
data parallelism and fault tolerance.

[Pyspark](https://spark.apache.org/docs/3.2.0/api/python/index.html) is the
Python API for Apache Spark, an open source, distributed computing framework and
set of libraries for real-time, large-scale data processing.

You can use pandera to validate `pyspark.sql.DataFrame` objects directly. First,
install `pandera` with the `pyspark` extra:

```bash
pip install 'pandera[pyspark]'
```

:::{note}
Pandera ships an optional
{ref}`Narwhals-powered backend <narwhals-backends>` that runs validation
against the native PySpark SQL execution path and shares its check
implementations with the Polars and Ibis backends. It is **opt-in**:
install the `narwhals` extra and set
`PANDERA_USE_NARWHALS_BACKEND=True` (or `pandera.config.CONFIG.use_narwhals_backend = True`)
before importing `pandera.pyspark`. By default Pandera uses the native PySpark
backend. The public API shown on this page is unchanged either way.

Because the Narwhals backend for PySpark shares its check implementations with
the Polars and Ibis backends, several behaviours differ from the native PySpark
backend:

- **SQL-lazy execution.** No element-wise checks (no `map_batches` on SQL-lazy
  frames), and no row sampling via `sample=` / `tail=` parameters.
- **`coerce=True` is a no-op.** The Narwhals `ColumnBackend` has no coercion
  step. Setting `coerce=True` on a `Field` or `Column` performs no coercion;
  Pandera emits a ``SchemaWarning`` per column to make the subsequent
  ``WRONG_DATATYPE`` error understandable rather than silent. Setting
  ``coerce=True`` at the `Config` level (row-wise `auto_coerce` dtype) is
  handled and does not warn.
  If you rely on `coerce=True` to convert column dtypes, use the native PySpark
  backend (`PANDERA_USE_NARWHALS_BACKEND=False`).
- **Custom checks using `PysparkDataframeColumnObject` are incompatible.**
  Custom checks registered via `@register_check_method` that expect a
  `pyspark_obj: PysparkDataframeColumnObject` argument will not work under the
  Narwhals backend. The Narwhals backend passes a `NarwhalsData(frame, key)`
  named tuple to check functions instead, so the custom check signature and
  body must be rewritten against the Narwhals frame API (or kept on the
  native backend).
- **`failure_cases` rows may be omitted for scalar Polars errors.** Schema-level
  failure cases produced as scalar Polars frames (e.g. from a wrong-dtype check)
  are still reported in the ``errors`` dict but their rows are omitted from the
  aggregated ``failure_cases`` frame. See the
  {ref}`Narwhals Known gaps <narwhals-backends>` section for details.
- **Unified `SchemaErrors` contract.** Like the Polars and Ibis Narwhals
  backends, the PySpark Narwhals backend raises `pandera.errors.SchemaErrors`
  on validation failure (or `SchemaError` for the first error when
  `lazy=False`). This differs from the native PySpark backend, which attaches
  errors to `dataframe.pandera.errors`. If you depend on the
  `dataframe.pandera.errors` accessor, use the native PySpark backend
  (`PANDERA_USE_NARWHALS_BACKEND=False`).

```bash
pip install 'pandera[pyspark,narwhals]'
export PANDERA_USE_NARWHALS_BACKEND=True
```
:::

## What's different?

Compared to the way `pandera` deals with pandas dataframes, there are some
small changes to support the nuances of pyspark SQL and the expectations that
users have when working with pyspark SQL dataframes:

1. The output of `schema.validate` will produce a dataframe in pyspark SQL
   even in case of errors during validation. Instead of raising the error, the
   errors are collected and can be accessed via the `dataframe.pandera.errors`
   attribute  as shown in this example.

   :::{note}
   This design decision is based on the expectation that most use cases for
   pyspark SQL dataframes means entails a production ETL setting. In these settings,
   pandera prioritizes completing the production load and saving the data quality
   issues for downstream rectification.
   :::

2. Unlike the pandera pandas schemas, the default behaviour of the pyspark SQL
   version for errors is `lazy=True`, i.e. all the errors would be collected
   instead of raising at first error instance.

3. There is no support for lambda based vectorized checks since in spark lambda
   checks needs UDFs, which is inefficient. However pyspark sql does support custom
   checks via the {func}`~pandera.extensions.register_check_method` decorator.

4. The custom check has to return a scalar boolean value instead of a series.

5. In defining the type annotation, there is limited support for default python
   data types such as `int`, `str`, etc. When using the `pandera.pyspark` API, using
   `pyspark.sql.types` based datatypes such as `StringType`, `IntegerType`,
   etc. is highly recommended.

## Basic Usage

In this section, lets look at an end to end example of how pandera would work in
a native pyspark implementation.

```{code-cell} python
import pandera.pyspark as pa
import pyspark.sql.types as T

from decimal import Decimal
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pandera.pyspark import DataFrameModel

spark = SparkSession.builder.getOrCreate()

class PanderaSchema(DataFrameModel):
    id: T.IntegerType() = pa.Field(gt=5)
    product_name: T.StringType() = pa.Field(str_startswith="B")
    price: T.DecimalType(20, 5) = pa.Field()
    description: T.ArrayType(T.StringType()) = pa.Field()
    meta: T.MapType(T.StringType(), T.StringType()) = pa.Field()

data = [
    (5, "Bread", Decimal(44.4), ["description of product"], {"product_category": "dairy"}),
    (15, "Butter", Decimal(99.0), ["more details here"], {"product_category": "bakery"}),
]

spark_schema = T.StructType(
    [
        T.StructField("id", T.IntegerType(), False),
        T.StructField("product", T.StringType(), False),
        T.StructField("price", T.DecimalType(20, 5), False),
        T.StructField("description", T.ArrayType(T.StringType(), False), False),
        T.StructField(
            "meta", T.MapType(T.StringType(), T.StringType(), False), False
        ),
    ],
)
df = spark.createDataFrame(data, spark_schema)
df.show()
```

In example above, the `PanderaSchema` class inherits from the `DataFrameModel` base
class. It has type annotations for 5 fields with 2 of the fields having checks
enforced e.g. `gt=5` and `str_startswith="B"`.

Just to simulate some schema and data validations, we also defined native spark's
schema `spark_schema` and enforced it on our dataframe `df`.

Next, you can use the {py:func}`~PanderaSchema.validate` function to validate
pyspark sql dataframes at runtime.

```{code-cell} python
df_out = PanderaSchema.validate(check_obj=df)
df_out
```

After running {py:func}`~PanderaSchema.validate`, the returned object `df_out`
will be a `pyspark` dataframe extended to hold validation results exposed via
a `pandera` attribute.

## Pandera Pyspark Error Report

*new in 0.16.0*

You can print the validation results as follows:

```{code-cell} python
import json

df_out_errors = df_out.pandera.errors
print(json.dumps(dict(df_out_errors), indent=4))
```

As seen above, the error report is aggregated on 2 levels in a python `dict` object:

1. The type of validation: `SCHEMA` or `DATA`
2. The category of errors such as `DATAFRAME_CHECK` or `WRONG_DATATYPE`, etc.

This error report is easily consumed by downstream applications such as timeseries
visualization of errors over time.

:::{important}
It's critical to extract errors report from `df_out.pandera.errors` as any
further `pyspark` operations may reset the attribute.
:::

## Granular Control of Pandera's Execution

*new in 0.16.0*

By default, error reports are generated for both schema and data level validation.
Adding support for pysqark SQL also comes with more granular control over the execution
of Pandera's validation flow.

This is achieved by introducing configurable settings using environment variables
that allow you to control execution at three different levels:

1. `SCHEMA_ONLY`: perform schema validations only. It checks that data conforms
   to the schema definition, but does not perform any data-level validations on dataframe.
2. `DATA_ONLY`: perform data-level validations only. It validates that data
   conforms to the defined `checks`, but does not validate the schema.
3. `SCHEMA_AND_DATA`: (**default**) perform both schema and data level
   validations. It runs most exhaustive validation and could be compute intensive.

You can override default behaviour by setting an environment variable from terminal
before running the `pandera` process as:

```bash
export PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY
```

This will be picked up by `pandera` to only enforce SCHEMA level validations.

## Switching Validation On and Off

*new in 0.16.0*

It's very common in production to enable or disable certain services to save
computing resources. We thought about it and thus introduced a switch to enable
or disable pandera in production.

You can override default behaviour by setting an environment variable from terminal
before running  the `pandera` process as follow:

```bash
export PANDERA_VALIDATION_ENABLED=False
```

This will be picked up by `pandera` to disable all validations in the application.

By default, validations are enabled and depth is set to `SCHEMA_AND_DATA` which
can be changed to `SCHEMA_ONLY` or `DATA_ONLY` as required by the use case.

## Caching control

*new in 0.17.3*

Given Spark's architecture and Pandera's internal implementation of PySpark integration
that relies on filtering conditions and *count* commands,
the PySpark DataFrame being validated by a Pandera schema may be reprocessed
multiple times, as each *count* command triggers a new underlying *Spark action*.
This processing overhead is directly related to the amount of *schema* and *data* checks
added to the Pandera schema.

To avoid such reprocessing time, Pandera allows you to cache the PySpark DataFrame
before validation starts, through the use of two environment variables:

```bash
export PANDERA_CACHE_DATAFRAME=True # Default is False, do not `cache()` by default
export PANDERA_KEEP_CACHED_DATAFRAME=True # Default is False, `unpersist()` by default
```

The first controls if current DataFrame state should be cached in your Spark Session
before the validation starts. The second controls if such cached state should still be
kept after the validation ends.

:::{note}
To cache or not is a trade-off analysis: if you have enough memory to keep
the dataframe cached, it will speed up the validation timings as the validation
process will make use of this cached state.

Keeping the cached state and opting for not throwing it away when the
validation ends is important when the Pandera validation of a dataset is not
an individual process, but one step of the pipeline: if you have a pipeline that,
in a single Spark session, uses Pandera to evaluate all input dataframes before
transforming them in an result that will be written to disk, it may make sense
to not throw away the cached states in this session. In the end, the already
processed states of these dataframes will still be used after the validation ends
and storing them in memory may be beneficial.
:::

## Registering Custom Checks

`pandera` already offers an interface to register custom checks functions so
that they're available in the {class}`~pandera.api.checks.Check` namespace. See
{ref}`the extensions document <extensions>` for more information.

Unlike the pandera pandas API, pyspark sql does not support lambda function inside `check`.
It is because to implement lambda functions would mean introducing spark UDF which
is expensive operation due to serialization, hence it is better to create native pyspark function.

Note: The output of the function should be a boolean value `True` for passed and
`False` for failure. Unlike the Pandas version which expect it to be a series
of boolean values.

```{code-cell} python
from pandera.extensions import register_check_method
import pyspark.sql.types as T
from pyspark.sql.functions import col

@register_check_method
def new_pyspark_check(pyspark_obj, *, max_value) -> bool:
    """Ensure values of the data are strictly below a maximum value.
    :param max_value: Upper bound not to be exceeded. Must be
        a type comparable to the dtype of the column datatype of pyspark
    """

    cond = col(pyspark_obj.column_name) <= max_value
    return pyspark_obj.dataframe.filter(~cond).limit(1).count() == 0

class Schema(DataFrameModel):
    """Schema"""

    product: T.StringType()
    code: T.IntegerType() = pa.Field(
        new_pyspark_check={
            "max_value": 30
        }
    )
```

## Adding Metadata at the Dataframe and Field level

*new in 0.16.0*

In real world use cases, we often need to embed additional information on objects.
Pandera that allows users to store additional metadata at `Field` and
`Schema` / `Model` levels. This feature is designed to provide greater context
and information about the data, which can be leveraged by other applications.

For example, by storing details about a specific column, such as data type, format,
or units, developers can ensure that downstream applications are able to interpret
and use the data correctly. Similarly, by storing information about which columns
of a schema are needed for a specific use case, developers can optimize data
processing pipelines, reduce storage costs, and improve query performance.

```{code-cell} python
import pyspark.sql.types as T

class PanderaSchema(DataFrameModel):
    """Pandera Schema Class"""

    product_id: T.IntegerType() = pa.Field()
    product_class: T.StringType() = pa.Field(
        metadata={
            "search_filter": "product_pricing",
        },
    )
    product_name: T.StringType() = pa.Field()
    price: T.DecimalType(20, 5) = pa.Field()

    class Config:
        """Config of pandera class"""

        name = "product_info"
        strict = True
        coerce = True
        metadata = {"category": "product-details"}
```

As seen in above example, `product_class` field has additional embedded information
such as `search_filter`. This metadata can be leveraged to search and filter
multiple schemas for certain keywords.

This is clearly a very basic example, but the possibilities are endless with having
metadata at `Field` and `` `DataFrame` `` levels.

We also provided a helper function to extract metadata from a schema as follows:

```{code-cell} python
PanderaSchema.get_metadata()
```

:::{note}
This feature is available for `pyspark.sql` and `pandas` both.
:::

## Embedding `Field` metadata in `Annotated`

You can also embed a {func}`~pandera.api.dataframe.model_components.Field`
directly inside {data}`typing.Annotated` to attach column-level metadata
— such as `description`, `title`, checks (`gt`, `ge`, `le`, `isin`, …),
or custom `metadata` — without providing an explicit `= pa.Field(...)`
assignment. This works for plain `pyspark.sql.types` as well as
parameterized ones:

```{code-cell} python
from typing import Annotated

import pyspark.sql.types as T


class ProductsModel(DataFrameModel):
    product_id: Annotated[T.IntegerType, pa.Field(title="Product ID")]
    product_name: Annotated[
        T.StringType, pa.Field(description="Product name")
    ]
    price: Annotated[T.DoubleType, pa.Field(gt=0.0, description="Unit price")]
    # parameterized dtypes can be combined with FieldInfo
    list_price: Annotated[
        T.DecimalType, 20, 5, pa.Field(description="Listed price")
    ]


schema = ProductsModel.to_schema()
assert schema.columns["product_name"].description == "Product name"
assert schema.columns["price"].checks
```

:::{note}
The `unique` argument is currently not supported in `Field(...)` for
PySpark schemas (see [`unique` support](#unique-support) below). Use
the model's `Config.unique` instead.
:::

## `unique` support

*new in 0.17.3*

:::{warning}
The `unique` support for PySpark-based validations to define which columns must be
tested for unique values may incur in a performance hit, given Spark's distributed
nature. It only works with `Config`.

Use with caution.
:::


## Supported and Unsupported Functionality

Since the pandera-pyspark-sql integration is less mature than pandas support, some
of the functionality offered by the pandera with pandas DataFrames are
not yet supported with pyspark sql DataFrames.

Here is a list of supported and unsupported features. You can
refer to the {ref}`supported features matrix <supported-features>` to see
which features are implemented in the pyspark-sql validation backend.
