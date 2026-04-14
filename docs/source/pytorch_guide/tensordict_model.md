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
    observation: torch.Tensor
    action: torch.Tensor

    @classmethod
    def _field(cls, field):
        return field(dtype=torch.float32, shape=(None, 10))
```

The `_field()` classmethod customizes the default field configuration.

## Validate with a model

```{code-cell} python
from tensordict import TensorDict

td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randn(32, 5)},
    batch_size=[32],
)
validated = RL.validate(td)
```

## Model configuration

Use {class}`~pandera.api.tensordict.model_components.Field` with the `_field()` method:

- `dtype`: Torch dtype
- `shape`: Expected shape tuple
- `checks`: List of Check instances
- `nullable`: Whether the key can be missing
- `default`: Default value if missing

```{code-cell} python
class RLWithConfig(pa.TensorDictModel):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor

    @classmethod
    def _field(cls, field):
        return field(
            dtype=torch.float32,
            shape=(None,),
            checks=[Check.greater_than(0.0)],
        )
```

## See also

- {ref}`pytorch-tensordict-schema` — dictionary-based schema
- {ref}`pytorch-checks` — value checks
- {ref}`configuration` — validation configuration