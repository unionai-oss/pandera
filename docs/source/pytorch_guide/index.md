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

## Define a schema with a class-based model

```{code-cell} python
class RL(pa.TensorDictModel):
    observation: torch.Tensor
    action: torch.Tensor

    @classmethod
    def _field(cls, field):
        return field(dtype=torch.float32, shape=(None, 10))

# Validate using the model
td = TensorDict(
    {"observation": torch.randn(32, 10), "action": torch.randn(32, 5)},
    batch_size=[32],
)
RL.validate(td)
```

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
