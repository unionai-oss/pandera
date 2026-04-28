---
file_format: mystnb
---

(pytorch-tensordict-io)=

# Serialization and Deserialization

Save and load TensorDict schemas using YAML or JSON, or save TensorDicts with embedded schema metadata.

## Schema Serialization

### YAML Format

```{code-cell} python
import torch
from pandera import Check
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
        "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
        "reward": pa.Tensor(dtype=torch.float32, shape=(None,), checks=Check.greater_than(0.0)),
    },
    batch_size=(32,),
)

# Save schema to YAML
schema_yaml = pa.to_yaml(schema)
print("YAML output preview:", schema_yaml[:150] + "...")

# Load schema from YAML
loaded_schema = pa.from_yaml(schema_yaml)
assert loaded_schema.batch_size == schema.batch_size
```

### JSON Format

```{code-cell} python
import json
import torch
from pandera import Check
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

# Serialize to JSON string
json_str = pa.to_json(schema)
print("JSON output preview:", json_str[:150] + "...")

# Deserialize from JSON string
loaded_schema = pa.from_json(json_str)
```

## File I/O

Save schemas to and load from files:

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

# Save to file
with tempfile.TemporaryDirectory() as tmpdir:
    schema_path = Path(tmpdir) / "schema.yaml"
    pa.to_yaml(schema, schema_path)
<<<<<<< HEAD
    
=======

>>>>>>> pr/pytorch-tensordict-phase3-4
    # Load from file
    loaded_schema = pa.from_yaml(schema_path)
    print("Successfully saved and loaded schema")
```

## TensorDict Saving/Loading

Save TensorDicts with embedded schema metadata for data integrity:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
        "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
    },
    batch_size=(32,),
)

td = TensorDict({
    "observation": torch.randn(32, 10),
    "action": torch.randint(0, 5, (32,)),
}, batch_size=[32])

# Save with schema metadata
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = f"{tmpdir}/rl_batch.pt"
    pa.save(schema, td, save_path)
<<<<<<< HEAD
    
=======

>>>>>>> pr/pytorch-tensordict-phase3-4
    # Load and validate automatically
    loaded_td = pa.load(save_path)
    print(f"Loaded batch size: {loaded_td.batch_size}")
```

## Use Cases

### 1. Data Pipeline Validation

Save trained model inputs/outputs with schema:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
import tempfile

# At training time
training_data = TensorDict({
    "input": torch.randn(100, 64),
    "target": torch.randint(0, 10, (100,)),
}, batch_size=[100])

schema = pa.TensorDictSchema(
    keys={
        "input": pa.Tensor(dtype=torch.float32, shape=(None, 64)),
        "target": pa.Tensor(dtype=torch.int64, shape=(None,)),
    },
    batch_size=(100,),
)

with tempfile.TemporaryDirectory() as tmpdir:
    pa.save(schema, training_data, f"{tmpdir}/training.pt")
<<<<<<< HEAD
    
=======

>>>>>>> pr/pytorch-tensordict-phase3-4
    # Later: validate before inference
    loaded = pa.load(f"{tmpdir}/training.pt")
    print("Successfully validated and loaded training data")
```

### 2. Configuration Management

Define schemas in YAML for version control and collaboration:

```{code-cell} python
import tempfile
from pathlib import Path
import pandera.tensordict as pa
import torch

# Define schema programmatically
schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
        "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
    },
    batch_size=(32,),
)

with tempfile.TemporaryDirectory() as tmpdir:
    # Save to config file
    config_path = Path(tmpdir) / "rl_schema.yaml"
    pa.to_yaml(schema, config_path)
<<<<<<< HEAD
    
=======

>>>>>>> pr/pytorch-tensordict-phase3-4
    # Load from config file (e.g., in a different process)
    loaded_schema = pa.from_yaml(config_path)
    print("Successfully loaded schema from config")
```

### 3. Distributed Training

Share schemas across workers via serialization:

```{code-cell} python
import torch
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

# Worker 1: Serialize and send
serialized = pa.to_json(schema)
print("Serialized schema (JSON):", serialized[:150] + "...")

# Worker 2: Deserialize and use (simulated here)
loaded_schema = pa.from_json(serialized)
print(f"Loaded batch_size: {loaded_schema.batch_size}")
```

## See also

- {ref}`pytorch-tensordict-inference` — infer schemas from data
- {ref}`pytorch-tensordict-schema` — manual schema definition
