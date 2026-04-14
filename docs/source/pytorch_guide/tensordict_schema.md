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

Set `lazy=True` to collect all errors instead of fail-fast:

```{code-cell} python
td_invalid = TensorDict(
    {"observation": torch.randn(16, 10), "action": torch.randn(32, 5)},
    batch_size=[16],
)

try:
    schema.validate(td_invalid, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(exc)
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

## See also

- {ref}`pytorch-tensordict-model` — class-based schema definition
- {ref}`pytorch-checks` — checks for value validation
- {ref}`pytorch-error-reporting` — error handling