# PyTorch Tensor Data Validation

[PyTorch](https://pytorch.org/) provides `torch.Tensor` objects,
`tensordict.TensorDict` for managing collections of tensors,
and [tensorclass](https://docs.pytorch.org/tensordict/stable/reference/tensorclass.html) for typed tensor collections.

Pandera validates them with the same patterns as the other dataframe backends:
schema objects, optional [`Check`](../reference/generated/pandera.api.checks.Check.html) instances, and
global [configuration](../configuration.html).

## Installation

```bash
pip install 'pandera[torch]'
pip install tensordict
```

## Quick start

```python
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

## Supported objects

[`TensorDictSchema`](../reference/generated/pandera.api.tensordict.container.TensorDictSchema.html) validates any
tensordict tensor collection:

- `tensordict.TensorDict` and any other
  `tensordict.TensorDictBase` subclass, including lazily stacked
  tensordicts created with `TensorDict.lazy_stack`.
- [tensorclass](https://docs.pytorch.org/tensordict/stable/reference/tensorclass.html) instances created with the
  `tensordict.tensorclass` decorator.

```python
stacked = TensorDict.lazy_stack(
    [
        TensorDict({"observation": torch.randn(16, 10), "action": torch.randn(16, 5)}, batch_size=[16]),
        TensorDict({"observation": torch.randn(16, 10), "action": torch.randn(16, 5)}, batch_size=[16]),
    ],
    dim=0,
)

stacked_schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32),
        "action": pa.Tensor(dtype=torch.float32),
    },
    batch_size=(2, 16),
)
stacked_schema.validate(stacked)
```

## Type Coercion

Set `coerce=True` to automatically convert tensor dtypes:

```python
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

```python
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
Use [`Field`](../reference/generated/pandera.api.tensordict.model_components.Field.html) to define additional constraints like shape and checks.

## Guide contents

- [TensorDictSchema](./tensordict_schema.html) — validating a `tensordict.TensorDict` with `Tensor` components
- [TensorDictModel](./tensordict_model.html) — class-based `TensorDictModel`
- [Checks and Lazy Validation](./tensordict_checks.html) — checks, parsers, and lazy validation
- [Schema Inference](./tensordict_schema_inference.html) — infer schemas from data automatically
- [Serialization and Deserialization](./tensordict_io.html) — save/load schemas with YAML/JSON
- [Hypothesis Strategies](./tensordict_strategies.html) — generate synthetic data with Hypothesis
- [Error Reporting](./error_reporting.html) — `SchemaError` / `SchemaErrors`, lazy validation, and failure cases

## See also

- [Supported DataFrame Libraries](../supported_libraries.html) — other backends
- [Checks](../checks.html) — general `Check` behaviour
- [Lazy Validation](../lazy_validation.html) — `lazy=True` and `SchemaErrors`
- [Configuration](../configuration.html) — `ValidationDepth` and environment variables
