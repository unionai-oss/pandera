---
file_format: mystnb
---

```{currentmodule} pandera
```

(scaling-fugue)=

# Data Validation with Fugue

Validation on big data comes in two forms. The first is performing one set of
validations on data that doesn't fit in memory. The second happens when a large dataset
is comprised of multiple groups that require different validations. In pandas semantics,
this would be the equivalent of a `groupby-validate` operation. This section will cover
using `pandera` for both of these scenarios.

`Pandera` has support for `Spark` and `Dask` DataFrames through `Modin` and
`PySpark Pandas`. Another option for running `pandera`  on top of native `Spark`
or `Dask` engines is [Fugue](https://github.com/fugue-project/fugue/) . `Fugue` is
an open source abstraction layer that ports `Python`, `pandas`, and `SQL` code to
`Spark` and `Dask`. Operations will be applied on DataFrames natively, minimizing
overhead.

## What is Fugue?

`Fugue` serves as an interface to distributed computing. Because of its non-invasive design,
existing `Python` code can be scaled to a distributed setting without significant changes.

To run the example, `Fugue` needs to installed separately. Using pip:

```bash
pip install 'fugue[spark]'
```

This will also install `PySpark` because of the `spark` extra. `Dask` is available
with the `dask` extra.

## Example

In this example, a pandas `DataFrame` is created with `state`, `city` and `price`
columns. `Pandera` will be used to validate that the `price` column values are within
a certain range.

```{code-cell} python
import pandas as pd

data = pd.DataFrame(
    {
        'state': ['FL','FL','FL','CA','CA','CA'],
        'city': [
            'Orlando', 'Miami', 'Tampa', 'San Francisco', 'Los Angeles', 'San Diego'
        ],
        'price': [8, 12, 10, 16, 20, 18],
    }
)
data
```

Validation is then applied using pandera. A `price_validation` function is
created that runs the validation. None of this will be new.

```{code-cell} python
import pandera.pandas as pa


price_check = pa.DataFrameSchema(
    {"price": pa.Column(int, pa.Check.in_range(min_value=5,max_value=20))}
)

def price_validation(data: pd.DataFrame) -> pd.DataFrame:
    return price_check.validate(data)
```

The `transform` function in `Fugue` is the easiest way to use `Fugue` with existing `Python`
functions as seen in the following code snippet. The first two arguments are the `DataFrame` and
function to apply. The keyword argument `schema` is required because schema is strictly enforced
in distributed settings. Here, the `schema` is simply `*` because no new columns are added.

The last part of the `transform` function is the `engine`. Here, a `SparkSession` object
is used to run the code on top of `Spark`. For Dask, users can pass a string `"dask"` or
can pass a Dask Client. Passing nothing uses the default pandas-based engine. Because we
passed a SparkSession in this example, the output is a Spark DataFrame.

```python
from fugue import transform
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark_df = transform(data, price_validation, schema="*", engine=spark)
spark_df.show()
```

```
+-----+-------------+-----+
|state|         city|price|
+-----+-------------+-----+
|   FL|      Orlando|    8|
|   FL|        Miami|   12|
|   FL|        Tampa|   10|
|   CA|San Francisco|   16|
|   CA|  Los Angeles|   20|
|   CA|    San Diego|   18|
+-----+-------------+-----+
```

## Validation by Partition

There is an interesting use case that arises with bigger datasets. Frequently, there are logical
groupings of data that require different validations. In the earlier sample data, the
price range for the records with `state` FL is lower than the range for the `state` CA.
Two {class}`~pandera.api.pandas.container.DataFrameSchema` will be created to reflect this. Notice their ranges
for the {class}`~pandera.api.checks.Check` differ.

```{code-cell} python
price_check_FL = pa.DataFrameSchema({
    "price": pa.Column(int, pa.Check.in_range(min_value=7,max_value=13)),
})

price_check_CA = pa.DataFrameSchema({
    "price": pa.Column(int, pa.Check.in_range(min_value=15,max_value=21)),
})

price_checks = {'CA': price_check_CA, 'FL': price_check_FL}
```

A slight modification is needed to our `price_validation` function. `Fugue` will partition
the whole dataset into multiple pandas `DataFrames`. Think of this as a `groupby`. By the
time `price_validation` is used, it only contains the data for one `state`. The appropriate
`DataFrameSchema` is pulled and then applied.

To partition our data by `state`, all we need to do is pass it into the `transform` function
through the `partition` argument. This splits up the data across different workers before they
each run the `price_validation` function. Again, this is like a groupby-validation.

```python
def price_validation(df: pd.DataFrame) -> pd.DataFrame:
    location = df['state'].iloc[0]
    check = price_checks[location]
    check.validate(df)
    return df

spark_df = transform(
    data,
    price_validation,
    schema="*",
    partition=dict(by="state"),
    engine=spark,
)

spark_df.show()
```

```
SparkDataFrame
state:str|city:str                                                 |price:long
---------+---------------------------------------------------------+----------
CA       |San Francisco                                            |16
CA       |Los Angeles                                              |20
CA       |San Diego                                                |18
FL       |Orlando                                                  |8
FL       |Miami                                                    |12
FL       |Tampa                                                    |10
Total count: 6
```

:::{note}
Because operations in a distributed setting are applied per partition, statistical
validators will be applied on each partition rather than the global dataset. If no
partitioning scheme is specified, `Spark` and `Dask` use default partitions. Be
careful about using operations like mean, min, and max without partitioning beforehand.

All row-wise validations scale well with this set-up.
:::

## Returning Errors

`Pandera` will raise a `SchemaError` by default that gets buried by the Spark error
messages. To return the errors as a DataFrame, we use can use the following approach. If
there are no errors in the data, it will just return an empty DataFrame.

To keep the errors for each partition, you can attach the partition key as a column in
the returned DataFrame.

```python
from pandera.errors import SchemaErrors

out_schema = "schema_context:str, column:str, check:str, \
check_number:int, failure_case:str, index:int"

out_columns = ["schema_context", "column", "check",
"check_number", "failure_case", "index"]

price_check = pa.DataFrameSchema(
    {"price": pa.Column(int, pa.Check.in_range(min_value=12,max_value=20))}
)

def price_validation(data:pd.DataFrame) -> pd.DataFrame:
    try:
        price_check.validate(data, lazy=True)
        return pd.DataFrame(columns=out_columns)
    except SchemaErrors as err:
        return err.failure_cases

transform(data, price_validation, schema=out_schema, engine=spark).show()
```

```
+--------------+------+----------------+------------+------------+-----+
|schema_context|column|           check|check_number|failure_case|index|
+--------------+------+----------------+------------+------------+-----+
|        Column| price|in_range(12, 20)|           0|           8|    0|
|        Column| price|in_range(12, 20)|           0|          10|    0|
+--------------+------+----------------+------------+------------+-----+
```
