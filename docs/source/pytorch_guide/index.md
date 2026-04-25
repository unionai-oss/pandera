---
file_format: mystnb
---

(pytorch-guide)=

# PyTorch Tensor Data Validation

[PyTorch](https://pytorch.org/) provides {class}`~torch.Tensor` objects and
[{class}`~tensordict.TensorDict` for managing collections of tensors],
and [{class}`~tensordict.tensorclass`][tensorclass] for typed tensor
collections.

Pandera validates them with the same patterns as the other dataframe backends:
schema objects, optional {class}`~pandera.api.checks.Check` instances, and
global {ref}`configuration <configuration>`.

## Installation

```bash
pip install 'pandera[torch]'
pip install tensordict
```

## Quick start

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

td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randn(32, 5)},
    batch_size=[32],
)
schema.validate(td)
```

## Type Coercion

Set `coerce=True` to automatically convert tensor dtypes:

```{code-cell} python
schema_coerce = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
    coerce=True,
)

# Input with wrong dtype (float64)
td_wrong_dtype = TensorDict(
    {"observation": torch.randn(32, 10).to(torch.float64)},
    batch_size=[32],
)

# Dtype is automatically coerced to float32
validated = schema_coerce.validate(td_wrong_dtype)
assert validated["observation"].dtype == torch.float32
```

## Define a schema with a class-based model

```{code-cell} python
class RL(pa.TensorDictModel):
    """Schema for reinforcement learning data."""

    # Use PyTorch dtypes in type annotations
    observation: torch.float32 = pa.Field(shape=(None, 10))
    action: torch.int64 = pa.Field(shape=(None,))
    reward: torch.float32 = pa.Field()

    class Config:
        batch_size = (32,)

# Validate using the model - schema is built automatically
td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randint(0, 4, (32,)), "reward": torch.randn(32)},
    batch_size=[32],
)
RL.validate(td)
```

**Note:** Type annotations specify the dtype (e.g., `torch.float32`, `torch.int64`).
Use {func}`~pandera.tensordict.Field` to define additional constraints like shape and checks.

## Guide contents

```{toctree}
:maxdepth: 2
:hidden:

tensordict_schema
tensordict_model
tensordict_checks
error_reporting
```

- {ref}`pytorch-tensordict-schema` — validating a {class}`~tensordict.TensorDict` with `Tensor` components
- {ref}`pytorch-tensordict-model` — class-based `TensorDictModel`
- {ref}`pytorch-checks` — checks, parsers, and lazy validation
- {ref}`pytorch-error-reporting` — `SchemaError` / `SchemaErrors`, lazy validation, and failure cases

## See also

- {ref}`supported-dataframe-libraries` — other backends
- {ref}`checks` — general `Check` behaviour
- {ref}`lazy-validation` — `lazy=True` and `SchemaErrors`
- {ref}`configuration` — `ValidationDepth` and environment variables
