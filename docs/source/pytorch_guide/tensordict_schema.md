---
file_format: mystnb
---

(pytorch-tensordict-schema)=

# TensorDictSchema

Use {class}`~pandera.api.tensordict.container.TensorDictSchema` to validate
{class}`~tensordict.TensorDict` objects.

## Define a schema

Define a dictionary-based schema with {class}`~pandera.api.tensordict.components.Tensor`
components:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
        "action": pa.Tensor(dtype=torch.float32, shape=(None, 5)),
    },
    batch_size=(32,),
)
```

- `keys`: A dictionary mapping key names to {class}`~pandera.api.tensordict.components.Tensor` objects
- `batch_size`: Required batch dimension sizes. Use `None` for flexible dimensions.

## Validate a TensorDict

```{code-cell} python
td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randn(32, 5)},
    batch_size=[32],
)
validated = schema.validate(td)
```

## Lazy validation

Set `lazy=True` to collect all errors instead of fail-fast. Note that TensorDict
validates batch dimensions on creation, so we need to use matching dimensions:

```{code-cell} python
td_wrong_dtype = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randint(0, 5, (32,))},
    batch_size=[32],
)

try:
    schema.validate(td_wrong_dtype, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(f"Total errors: {len(exc.schema_errors)}")
    for err in exc.schema_errors:
        print(f"  - {str(err)}")
```

## Tensor component

The {class}`~pandera.api.tensordict.components.Tensor` component validates individual tensor keys.

### Parameters

- `dtype`: Torch dtype (e.g., `torch.float32`, `torch.int64`)
- `shape`: Expected shape tuple. Use `None` for flexible dimensions.
- `checks`: Optional list of {class}`~pandera.api.checks.Check` instances

### Example

```{code-cell} python
from pandera import Check

schema = pa.TensorDictSchema(
    keys={
        "values": pa.Tensor(
            dtype=torch.float32,
            shape=(None,),
            checks=[Check.greater_than(0.0), Check.less_than(1.0)],
        ),
    },
    batch_size=(10,),
)
```

## Type Coercion

Set `coerce=True` to automatically convert tensor dtypes during validation:

```{code-cell} python
schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
        "action": pa.Tensor(dtype=torch.int64),
    },
    batch_size=(32,),
    coerce=True,
)

# Input has wrong dtypes (float64, int32)
td = TensorDict(
    {
        "observation": torch.randn(32, 10).to(torch.float64),
        "action": torch.randint(0, 5, (32,)).to(torch.int32),
    },
    batch_size=[32],
)

# Dtypes are automatically coerced to float32 and int64
validated = schema.validate(td)
assert validated["observation"].dtype == torch.float32
assert validated["action"].dtype == torch.int64
```

Type coercion is applied **before** validation checks, so any dtype or shape
constraints will be evaluated on the coerced data.

## See also

- {ref}`pytorch-tensordict-model` — class-based schema definition
- {ref}`pytorch-checks` — checks for value validation
- {ref}`pytorch-error-reporting` — error handling
