---
file_format: mystnb
---

```{currentmodule} pandera
```

(pydantic-integration)=

# Pydantic

*new in 0.8.0*

## Using Pandera Schemas in Pydantic Models

{class}`~pandera.api.pandas.model.DataFrameModel` is fully compatible with
[pydantic](https://pydantic-docs.helpmanual.io/). You can specify
a {class}`~pandera.api.pandas.model.DataFrameModel` in a pydantic `BaseModel` as you would
any other field:

```{code-cell} python
:tags: [raises-exception]

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
import pydantic


class SimpleSchema(pa.DataFrameModel):
    str_col: Series[str] = pa.Field(unique=True)


class PydanticModel(pydantic.BaseModel):
    x: int
    df: DataFrame[SimpleSchema]


valid_df = pd.DataFrame({"str_col": ["hello", "world"]})
PydanticModel(x=1, df=valid_df)

invalid_df = pd.DataFrame({"str_col": ["hello", "hello"]})
PydanticModel(x=1, df=invalid_df)
```

Other pandera components are also compatible with pydantic:

:::{note}
The `SeriesSchema`, `DataFrameSchema` and `schema_components` types
validates the type of a schema object, e.g. if your pydantic
`BaseModel` contained a schema object, not a `pandas` object.
:::

- {class}`~pandera.api.pandas.model.DataFrameModel`
- {class}`~pandera.api.pandas.container.DataFrameSchema`
- {class}`~pandera.api.pandas.array.SeriesSchema`
- {class}`~pandera.api.pandas.components.MultiIndex`
- {class}`~pandera.api.pandas.components.Column`
- {class}`~pandera.api.pandas.components.Index`

## Using Pydantic Models in Pandera Schemas

*new in 0.10.0*

You can also use a pydantic `BaseModel` in a pandera schema. Suppose you had
a `Record` model:

```{code-cell} python
from pydantic import BaseModel

import pandera.pandas as pa


class Record(BaseModel):
    name: str
    xcoord: int
    ycoord: int
```

The {class}`~pandera.pandas_engine.PydanticModel` datatype enables you to
specify the `Record` model as a row-wise type.

```{code-cell} python
import pandas as pd
from pandera.engines.pandas_engine import PydanticModel


class PydanticSchema(pa.DataFrameModel):
    """Pandera schema using the pydantic model."""

    class Config:
        """Config with dataframe-level data type."""

        dtype = PydanticModel(Record)
        coerce = True  # this is required, otherwise a SchemaInitError is raised
```

:::{note}
By combining `dtype=PydanticModel(...)` and `coerce=True`, pandera will
apply the pydantic model validation process to each row of the dataframe,
converting the model back to a dictionary with the `BaseModel.dict()` method.
:::

The equivalent pandera schema would look like this:

```{code-cell} python
class PanderaSchema(pa.DataFrameModel):
    """Pandera schema that's equivalent to PydanticSchema."""

    name: pa.typing.Series[str]
    xcoord: pa.typing.Series[int]
    ycoord: pa.typing.Series[int]
```

:::{note}
Since the {class}`~pandera.pandas_engine.PydanticModel` datatype
applies the `BaseModel` constructor to each row of the dataframe, using
`PydanticModel` might not scale well with larger datasets.

**If you want to help benchmark**, consider
[contributing a benchmark script](https://github.com/pandera-dev/pandera/issues/794)
:::
