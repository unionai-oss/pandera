---
file_format: mystnb
---

(pytorch-tensordict-model)=

# TensorDictModel

Use {class}`~pandera.api.tensordict.model.TensorDictModel` to define a class-based
schema that maps directly to a {class}`~tensordict.TensorDict`.

## Define a model

```{code-cell} python
import torch
import pandera.tensordict as pa

class RL(pa.TensorDictModel):
    """Schema for reinforcement learning data."""

    # Type annotation specifies the dtype (torch.float32, torch.int64, etc.)
    observation: torch.float32 = pa.Field(shape=(None, 10))
    action: torch.int64 = pa.Field(shape=(None,))
    reward: torch.float32 = pa.Field()
```

Use PyTorch dtypes in type annotations (e.g., `torch.float32`, `torch.int64`)
to specify the expected data type. Use {func}`~pandera.tensordict.Field` to
define additional constraints.

## Validate with a model

```{code-cell} python
from tensordict import TensorDict

td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randint(0, 4, (32,)), "reward": torch.randn(32)},
    batch_size=[32],
)
validated = RL.validate(td)
```

## Field configuration

Use {func}`~pandera.tensordict.Field` to customize field options:

- `shape`: Expected shape tuple (use `None` for variable dimensions)
- `checks`: List of Check instances or check arguments
- `nullable`: Whether the key can be missing
- `default`: Default value if missing

```{code-cell} python
class RLWithConfig(pa.TensorDictModel):
    """RL schema with field-level checks."""

    observation: torch.float32 = pa.Field(
        shape=(None, 10),
        ge=-1.0,
        le=1.0,
    )
    action: torch.int64 = pa.Field(
        shape=(None,),
        isin=[0, 1, 2, 3],
    )
    reward: torch.float32 = pa.Field(
        gt=0.0,
    )

    class Config:
        batch_size = (32,)
```

## Configuring the model

Use a nested `Config` class to configure schema-level options:

- `batch_size`: Expected batch size tuple (use `None` for variable dimensions)

```{code-cell} python
class RLWithBatchSize(pa.TensorDictModel):
    observation: torch.float32 = pa.Field(shape=(None, 10))
    action: torch.int64 = pa.Field(shape=(None,))

    class Config:
        batch_size = (64,)
```

## Lazy validation

Use `lazy=True` to collect all validation errors:

```{code-cell} python
from tensordict import TensorDict

# Create invalid data with wrong dtypes
td_wrong_dtype = TensorDict(
    {"observation": torch.randn(32, 10).to(torch.float64), "action": torch.randint(0, 4, (32,)), "reward": torch.randn(32)},
    batch_size=[32],
)

try:
    RL.validate(td_wrong_dtype, lazy=True)
except pa.SchemaErrors as e:
    print(f"Found {len(e.schema_errors)} validation errors:")
    for err in e.schema_errors:
        print(f"  - {err.reason_code}")
```

## Type Coercion

Model schemas support dtype coercion with the `coerce=True` option:

```{code-cell} python
class RLWithCoercion(pa.TensorDictModel):
    observation: torch.float32 = pa.Field(shape=(None, 10))
    
    class Config:
        batch_size = (32,)
        coerce = True

# Input with wrong dtype
td = TensorDict(
    {"observation": torch.randn(32, 10).to(torch.float64)},
    batch_size=[32],
)

# Dtype is automatically coerced during validation
validated = RLWithCoercion.validate(td)
assert validated["observation"].dtype == torch.float32
```

## Model inheritance

Models can be inherited to create more specific schemas:

```{code-cell} python
class BaseRL(pa.TensorDictModel):
    observation: torch.float32 = pa.Field(shape=(None, 10))

    class Config:
        batch_size = (32,)

class ExtendedRL(BaseRL):
    action: torch.int64 = pa.Field(shape=(None,))

# ExtendedRL has both 'observation' and 'action'
schema = ExtendedRL.to_schema()
```

## See also

- {ref}`pytorch-tensordict-schema` — dictionary-based schema
- {ref}`pytorch-checks` — value checks
- {ref}`configuration` — validation configuration
