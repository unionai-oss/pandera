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
from pandera import Check

class RL(pa.TensorDictModel):
    observation: pa.DataType
    action: pa.DataType
```

Use {class}`~pandera.tensordict.DataType` as the type annotation and
{func}`~pandera.tensordict.Field` to customize field options.

## Validate with a model

```{code-cell} python
from tensordict import TensorDict

td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randn(32, 5)},
    batch_size=[32],
)
validated = RL.validate(td)
```

## Field configuration

Use {func}`~pandera.tensordict.Field` to customize field options:

- `dtype`: Torch dtype
- `shape`: Expected shape tuple
- `checks`: List of Check instances
- `nullable`: Whether the key can be missing
- `default`: Default value if missing

```{code-cell} python
class RLWithConfig(pa.TensorDictModel):
    observation: pa.DataType = pa.Field(dtype=torch.float32, shape=(None, 10))
    action: pa.DataType = pa.Field(dtype=torch.float32, shape=(None, 5))
    reward: pa.DataType = pa.Field(
        dtype=torch.float32,
        shape=(None,),
        checks=[Check.greater_than(0.0)],
    )
```

## See also

- {ref}`pytorch-tensordict-schema` — dictionary-based schema
- {ref}`pytorch-checks` — value checks
- {ref}`configuration` — validation configuration