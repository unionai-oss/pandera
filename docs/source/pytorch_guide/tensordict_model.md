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

## Custom checks in model fields

Model fields accept one-off :class:`~pandera.api.checks.Check` objects
through the `checks` argument, so a schema with a custom check (a valid
one-hot row, an ignore-index label, ...) no longer has to fall back to the
object API.

```{code-cell} python
from pandera import Check

def is_one_hot(t):
    return (t.sum(dim=-1) == 1.0) & ((t == 0) | (t == 1)).all(dim=-1)

class Policy(pa.TensorDictModel):
    probs: torch.float32 = pa.Field(
        shape=(None, 3),
        checks=Check(is_one_hot, name="one_hot", element_wise=False),
    )

    class Config:
        batch_size = (3,)

valid = TensorDict(
    {"probs": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
    batch_size=[3],
)
Policy.validate(valid)
```

Intrinsic checks (like `ge`, `le`, `isin`) can be combined with explicit
`Check` objects on the same field.

## Nested (tuple) keys in models

TensorDicts can nest, and TorchRL uses that for keys like
`("next", "observation")`. A model field maps to a nested key with
`pa.Field(alias=...)`:

```{code-cell} python
class NestedRL(pa.TensorDictModel):
    next_observation: torch.float32 = pa.Field(
        shape=(None, 10),
        alias=("next", "observation"),
    )
    next_reward: torch.float32 = pa.Field(
        shape=(None,),
        alias=("next", "reward"),
    )

    class Config:
        batch_size = (4,)

nested = TensorDict(
    {
        "next": TensorDict(
            {
                "observation": torch.randn(4, 10),
                "reward": torch.randn(4),
            },
            batch_size=[4],
        ),
    },
    batch_size=[4],
)
NestedRL.validate(nested)
```

## `@check_types` integration

The {func}`~pandera.decorators.check_types` decorator validates function
arguments and return values annotated with `TensorDict[Model]`:

```{code-cell} python
from pandera import check_types
from pandera.typing.tensordict import TensorDict as TDT

@check_types
def rl_step(td: TDT[RL]) -> TDT[RL]:
    return td

# Invalid input (wrong dtype) is caught at the function boundary
bad = TensorDict(
    {
        "observation": torch.randn(32, 10).to(torch.float64),
        "action": torch.randint(0, 4, (32,)),
        "reward": torch.randn(32),
    },
    batch_size=[32],
)

try:
    rl_step(bad)
except pa.errors.SchemaError as exc:
    print(f"check_types rejected input: {exc.reason_code}")
```

## See also

- {ref}`pytorch-tensordict-schema` — dictionary-based schema
- {ref}`pytorch-checks` — value checks
- {ref}`configuration` — validation configuration
