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

## Schema Inference

Automatically infer a schema from existing data using `pa.infer_schema()`:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

# Existing TensorDict with sample data
sample_td = TensorDict({
    "observation": torch.randn(100, 64),
    "action": torch.randint(0, 4, (100,)),
}, batch_size=[100])

# Infer schema from data
inferred_schema = pa.infer_schema(sample_td)
```

See {ref}`pytorch-tensordict-inference` for more details on schema inference.

## Serialization

Save and load schemas using YAML or JSON:

```{code-cell} python
import tempfile
from pathlib import Path
import torch
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

with tempfile.TemporaryDirectory() as tmpdir:
    # Save to YAML
    yaml_path = Path(tmpdir) / "schema.yaml"
    pa.to_yaml(schema, yaml_path)
    
    # Load from YAML
    loaded_schema = pa.from_yaml(yaml_path)
```

See {ref}`pytorch-tensordict-io` for more details on serialization.

## See also

- {ref}`pytorch-tensordict-model` — class-based schema definition
- {ref}`pytorch-checks` — checks for value validation
- {ref}`pytorch-error-reporting` — error handling
- {ref}`pytorch-tensordict-inference` — infer schemas from data
- {ref}`pytorch-tensordict-io` — save/load schemas
