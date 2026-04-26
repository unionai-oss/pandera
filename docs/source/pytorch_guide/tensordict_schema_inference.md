---
file_format: mystnb
---

(pytorch-tensordict-inference)=

# Schema Inference

Use `pa.infer_schema()` to automatically infer a schema from existing TensorDict data.

```{note}
Schema inference is useful when you have existing data and want to create a validation schema without manually specifying all the dtypes, shapes, and value ranges.
```

## Basic Usage

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

# Existing data with batch_size (100,)
td = TensorDict({
    "observation": torch.randn(100, 64),
    "action": torch.randint(0, 4, (100,)),
    "reward": torch.randn(100),
}, batch_size=[100])

# Infer schema from data
schema = pa.infer_schema(td)
print(f"Keys: {list(schema.keys.keys())}")
print(f"Batch size: {schema.batch_size}")

# Output includes inferred dtypes and shapes:
# TensorDictSchema(keys={'observation', 'action', 'reward'}, batch_size=(100,))
```

## What Gets Inferred

- **Dtypes**: Automatically detected (e.g., `torch.float32`, `torch.int64`)
- **Shapes**: Full tensor shapes including batch dimensions
- **Value ranges**: Min/max bounds for numeric tensors (used in range checks)

## TensorClass Support

Works with both `TensorDict` and `tensorclass`:

```{code-cell} python
from tensordict import tensorclass

@tensorclass
class RLData:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor

tc = RLData(
    observation=torch.randn(100, 32),
    action=torch.randint(0, 4, (100,)),
    reward=torch.randn(100),
    batch_size=[100]
)

# Infer schema from tensorclass
schema = pa.infer_schema(tc)
print(f"Keys: {list(schema.keys.keys())}")
```

## Use Cases

### 1. Quick Schema Development

Start with sample data and infer the schema, then refine it:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
from pandera import Check

# Step 1: Infer from sample data (batch_size=32)
sample_td = TensorDict({
    "observation": torch.randn(32, 10),
    "action": torch.randint(0, 4, (32,)),
}, batch_size=[32])
schema = pa.infer_schema(sample_td)

# Step 2: Refine the inferred schema for your needs
schema.keys["observation"].checks.append(Check.greater_than_or_equal_to(-1.0))
```

### 2. Data Quality Validation

Infer a baseline schema from healthy data, then validate new data against it:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

# Infer from known-good dataset (batch_size=100)
healthy_data = TensorDict({
    "observation": torch.randn(100, 64),
}, batch_size=[100])
baseline_schema = pa.infer_schema(healthy_data)

# Validate incoming batches with matching batch size
new_batch = TensorDict({"observation": torch.randn(100, 64)}, batch_size=[100])
validated_batch = baseline_schema.validate(new_batch)
```

### 3. Schema Evolution

Track how schemas change over time by inferring and comparing:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

# Infer from different dataset versions (both with batch_size=10)
data_v1 = TensorDict({"x": torch.randn(10)}, batch_size=[10])
schema_v1 = pa.infer_schema(data_v1)

data_v2 = TensorDict({"x": torch.randn(10), "y": torch.randn(10)}, batch_size=[10])
schema_v2 = pa.infer_schema(data_v2)

# Compare key sets
assert set(schema_v2.keys.keys()) == {"x", "y"}
```

## See also

- {ref}`pytorch-tensordict-schema` — manual schema definition
- {ref}`pytorch-tensordict-io` — saving and loading schemas
