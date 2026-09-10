---
file_format: mystnb
---

(xarray-data-tree)=

# DataTree Validation

{class}`~xarray.DataTree` is a hierarchical tree of datasets, useful for
organising related multi-dimensional data under a single structure.  Pandera
validates trees with {class}`~pandera.api.xarray.container.DataTreeSchema`
(imperative API) and {class}`~pandera.api.xarray.model.DataTreeModel`
(declarative API).

## `DataTreeSchema`

### Basic usage

`DataTreeSchema` validates node-level attributes and child nodes.  Each child
schema can be a {class}`~pandera.api.xarray.container.DatasetSchema` or another
`DataTreeSchema` for recursive nesting.

```{code-cell} python
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DataTreeSchema(
    attrs={"conventions": "CF-1.8"},
    children={
        "surface": pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
            },
        ),
        "upper": pa.DatasetSchema(
            data_vars={
                "wind": pa.DataVar(dtype=np.float64, dims=("x",)),
            },
        ),
    },
)

dt = xr.DataTree.from_dict({
    "/": xr.Dataset(attrs={"conventions": "CF-1.8"}),
    "/surface": xr.Dataset(
        {"temperature": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    ),
    "/upper": xr.Dataset(
        {"wind": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    ),
})
schema.validate(dt)
```

### Path-based children

Children can reference nested nodes using `/`-separated paths, just like
{meth}`xr.DataTree.from_dict`:

```{code-cell} python
nested_dt = xr.DataTree.from_dict({
    "/": xr.Dataset(attrs={"conventions": "CF-1.8"}),
    "/surface": xr.Dataset(
        {"temperature": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    ),
    "/surface/diagnostics": xr.Dataset(
        {"rmse": (("x",), np.ones(3, dtype=np.float64))},
        coords={"x": np.arange(3, dtype=np.float64)},
    ),
})

schema = pa.DataTreeSchema(
    children={
        "surface/diagnostics": pa.DatasetSchema(
            data_vars={"rmse": pa.DataVar(dtype=np.float64)},
        ),
    },
)
schema.validate(nested_dt)
```

### Root node dataset

Use the `dataset` parameter to validate the dataset attached to the root node:

```{code-cell} python
schema = pa.DataTreeSchema(
    dataset=pa.DatasetSchema(attrs={"conventions": "CF-1.8"}),
    children={
        "surface": pa.DatasetSchema(
            data_vars={
                "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
            },
        ),
    },
)
schema.validate(dt)
```

### Strict mode

When `strict=True`, unexpected child nodes raise a validation error:

```{code-cell} python
schema = pa.DataTreeSchema(
    children={
        "surface": pa.DatasetSchema(),
        "upper": pa.DatasetSchema(),
    },
    strict=True,
)
schema.validate(dt)
```

```{code-cell} python
strict_schema = pa.DataTreeSchema(
    children={"surface": pa.DatasetSchema()},
    strict=True,
)

try:
    strict_schema.validate(dt)
except pa.errors.SchemaError as exc:
    print(exc)
```

### Nested `DataTreeSchema`

Children can themselves be `DataTreeSchema` instances for deep validation:

```{code-cell} python
schema = pa.DataTreeSchema(
    attrs={"conventions": "CF-1.8"},
    children={
        "surface": pa.DataTreeSchema(
            dataset=pa.DatasetSchema(
                data_vars={
                    "temperature": pa.DataVar(dtype=np.float64, dims=("x",)),
                },
            ),
            children={
                "diagnostics": pa.DatasetSchema(
                    data_vars={"rmse": pa.DataVar(dtype=np.float64)},
                ),
            },
        ),
    },
)
schema.validate(nested_dt)
```

## `DataTreeModel`

### Basic usage

{class}`~pandera.api.xarray.model.DataTreeModel` uses class attributes
annotated with {class}`~pandera.api.xarray.model.DatasetModel` subclasses to
declare child node schemas:

```{code-cell} python
from pandera.typing.xarray import Coordinate

class SurfaceModel(pa.DatasetModel):
    temperature: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]

class UpperModel(pa.DatasetModel):
    wind: np.float64 = pa.Field(dims=("x",))
    x: Coordinate[np.float64]

class ClimateTree(pa.DataTreeModel):
    surface: SurfaceModel
    upper: UpperModel

    class Config:
        strict = True

ClimateTree.validate(dt)
```

### `Config` options

`DataTreeModel.Config`
({class}`~pandera.api.xarray.model_config.DataTreeConfig`) accepts:
`strict`, `attrs`, `name`.

### Field name access

```{code-cell} python
print(ClimateTree.surface)
print(ClimateTree.upper)
```

### `to_schema()` and `validate()`

```{code-cell} python
schema = ClimateTree.to_schema()
print(type(schema))

ClimateTree.validate(dt)
```

### `@check_types` with `DataTree`

Use `DataTree[Model]` from `pandera.typing.xarray` with the `@check_types`
decorator:

```{code-cell} python
from pandera.typing.xarray import DataTree

@pa.check_types
def process_tree(tree: DataTree[ClimateTree]) -> DataTree[ClimateTree]:
    return tree

process_tree(dt)
```

## See also

- {ref}`xarray-data-array-schema` / {ref}`xarray-dataset-schema` — imperative API
- {ref}`xarray-data-models` — `DataArrayModel` and `DatasetModel`
- {ref}`xarray-decorators` — decorator-based validation
