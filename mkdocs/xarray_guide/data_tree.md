# DataTree Validation

`xarray.DataTree` is a hierarchical tree of datasets, useful for
organising related multi-dimensional data under a single structure.  Pandera
validates trees with [`DataTreeSchema`](../reference/generated/pandera.api.xarray.container.DataTreeSchema.html)
(imperative API) and [`DataTreeModel`](../reference/generated/pandera.api.xarray.model.DataTreeModel.html)
(declarative API).

## `DataTreeSchema`

### Basic usage

`DataTreeSchema` validates node-level attributes and child nodes.  Each child
schema can be a [`DatasetSchema`](../reference/generated/pandera.api.xarray.container.DatasetSchema.html) or another
`DataTreeSchema` for recursive nesting.

```python
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
`xr.DataTree.from_dict`:

```python
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

```python
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

```python
schema = pa.DataTreeSchema(
    children={
        "surface": pa.DatasetSchema(),
        "upper": pa.DatasetSchema(),
    },
    strict=True,
)
schema.validate(dt)
```

```python
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

```python
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

[`DataTreeModel`](../reference/generated/pandera.api.xarray.model.DataTreeModel.html) uses class attributes
annotated with [`DatasetModel`](../reference/generated/pandera.api.xarray.model.DatasetModel.html) subclasses to
declare child node schemas:

```python
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
([`DataTreeConfig`](../reference/generated/pandera.api.xarray.model_config.DataTreeConfig.html)) accepts:
`strict`, `attrs`, `name`.

### Field name access

```python
print(ClimateTree.surface)
print(ClimateTree.upper)
```

### `to_schema()` and `validate()`

```python
schema = ClimateTree.to_schema()
print(type(schema))

ClimateTree.validate(dt)
```

### `@check_types` with `DataTree`

Use `DataTree[Model]` from `pandera.typing.xarray` with the `@check_types`
decorator:

```python
from pandera.typing.xarray import DataTree

@pa.check_types
def process_tree(tree: DataTree[ClimateTree]) -> DataTree[ClimateTree]:
    return tree

process_tree(dt)
```

## See also

- [DataArray Schemas](./data_array_schema.html) / [DatasetSchema](./dataset_schema.html) — imperative API
- [Data Models](./data_models.html) — `DataArrayModel` and `DatasetModel`
- [Decorators](./decorators.html) — decorator-based validation
