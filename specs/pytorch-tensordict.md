# PyTorch TensorDict Integration Spec

> **Status:** Draft
> **Issue:** TBD
> **Author:** opencode
> **Related work:** [xarray-schema](https://github.com/xarray-contrib/xarray-schema)

---

## 1. Motivation

[TensorDict](https://pytorch.org/tensordict/) is a powerful container for managing groups of tensors with the same batch dimension. It is central to many modern PyTorch workflows, especially in reinforcement learning and large-scale distributed training.

Adding TensorDict support extends pandera's reach to the PyTorch ecosystem, providing a consistent validation API for high-dimensional tensor containers, alongside its existing tabular backends.

### The Problem: Imperative Validation is Brittle

In real-world ML pipelines, tensor containers often need to satisfy complex constraints that go beyond structural integrity:

- Ensuring tensor values fall within a specific range (e.g., normalized inputs between -1 and 1).
- Validating that specific keys exist with expected dtypes.
- Checking that batch dimensions are consistent across all tensors.
- Ensuring data meets quality standards before expensive training runs.

While `tensordict` ensures that all tensors share a common `batch_size`, it doesn't provide built-in mechanisms for validating the **content** or **metadata** of those tensors. Developers often resort to verbose, imperative validation code that is hard to maintain and reuse.

### The Solution: Declarative Schemas with Pandera

Pandera adds value by enabling **declarative, schema-based validation** for TensorDict and tensorclass objects:

1.  **Type Safety:** Catch structural errors at development time with Python type hints via `@check_types`.
2.  **Complex Value Checks:** Leverage the existing Pandera `Check` ecosystem (e.g., `in_range`, `isin`, `greater_than`, etc.) for tensor values and dimensions.
3.  **Unified API:** Use the same validation patterns across pandas, polars, and PyTorch for a consistent developer experience.
4.  **Reusable Schemas:** Define validation logic once in a schema, then reuse it across multiple pipelines or data sources.

---

### Comprehensive Usage Examples

#### Example 1: Validating TensorDict Structure and Values

**Before (Imperative Validation - The "Pants on Fire" Approach)**

```python
import torch
from tensordict import TensorDict

def validate_batch(td: TensorDict):
    """Manual validation - error-prone, hard to maintain."""
    # Check batch size
    assert td.batch_size[0] == 32, f"Expected batch_size[0]=32, got {td.batch_size[0]}"
    
    # Check keys exist
    assert "observation" in td, "Missing 'observation' key"
    assert "action" in td, "Missing 'action' key"
    
    # Check dtypes
    assert td["observation"].dtype == torch.float32, f"Expected float32, got {td['observation'].dtype}"
    assert td["action"].dtype == torch.float32, f"Expected float32, got {td['action'].dtype}"
    
    # Check value ranges (normalized observations)
    obs = td["observation"]
    assert obs.min() >= -1.0, f"observation min value {obs.min()} < -1.0"
    assert obs.max() <= 1.0, f"observation max value {obs.max()} > 1.0"
    
    # Check action bounds
    action = td["action"]
    assert action.min() >= -2.0, f"action min value {action.min()} < -2.0"
    assert action.max() <= 2.0, f"action max value {action.max()} > 2.0"

# Usage
batch = TensorDict({
    "observation": torch.rand(32, 10) * 2 - 1,  # Range [-1, 1]
    "action": torch.rand(32, 5) * 4 - 2           # Range [-2, 2]
}, batch_size=[32])

validate_batch(batch)  # Runs all checks
```

**After (Declarative Validation with Pandera)**

```python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
from pandera import Check

# Define the schema once - reusable and self-documenting
batch_schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(
            dtype=torch.float32,
            shape=(32, 10),
            checks=[
                Check.greater_than_or_equal_to(-1.0),
                Check.less_than_or_equal_to(1.0),
            ]
        ),
        "action": pa.Tensor(
            dtype=torch.float32,
            shape=(32, 5),
            checks=[
                Check.greater_than_or_equal_to(-2.0),
                Check.less_than_or_equal_to(2.0),
            ]
        ),
    },
    batch_size=(32,)  # Validate batch_size[0] == 32
)

# Usage - clean and simple
batch = TensorDict({
    "observation": torch.rand(32, 10) * 2 - 1,
    "action": torch.rand(32, 5) * 4 - 2
}, batch_size=[32])

validated_batch = batch_schema.validate(batch)
```

#### Example 2: Lazy Validation with Multiple Errors

When validation fails, you want to see **all** errors at once, not just the first one:

```python
import pandera.tensordict as pa
from pandera import Check, SchemaError

# Schema with multiple checks per tensor
policy_schema = pa.TensorDictSchema(
    keys={
        "logits": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 10),  # None means "any size"
            checks=Check.in_range(-10.0, 10.0)
        ),
        "values": pa.Tensor(
            dtype=torch.float32,
            shape=(None,),
            checks=Check.greater_than(0.0)  # Value estimates must be positive
        ),
    },
    batch_size=(64,)
)

# Create invalid data
invalid_batch = TensorDict({
    "logits": torch.randn(64, 10) * 20,  # Some values outside [-10, 10]
    "values": torch.randn(64)             # Some negative values
}, batch_size=[64])

# Validate with lazy=True to collect ALL errors
try:
    policy_schema.validate(invalid_batch, lazy=True)
except pa.SchemaErrors as e:
    print(e)
    # Output shows both dtype and value errors for each key
```

#### Example 3: Declarative API with `TensorDictModel`

For a more Pydantic-like experience, use class-based models:

```python
import torch
import pandera.tensordict as pa
from pandera import Field, Check

class PolicyModel(pa.TensorDictModel):
    """Schema for a policy network output."""
    
    logits: torch.Tensor = Field(
        dtype=torch.float32,
        shape=(None, 10),
        checks=Check.in_range(-5.0, 5.0)
    )
    value: torch.Tensor = Field(
        dtype=torch.float32,
        shape=(None,),
        checks=Check.greater_than(0.0)
    )
    
    class Config:
        batch_size = (64,)

# Validate using the model
policy_model = PolicyModel()
batch = TensorDict({
    "logits": torch.randn(64, 10),
    "value": torch.randn(64).abs()  # Ensure positive
}, batch_size=[64])

validated = policy_model.validate(batch)
```

#### Example 4: Validating `tensorclass` Objects

The same schema works for both `TensorDict` and `tensorclass`:

```python
from tensordict import TensorDict, tensorclass
import pandera.tensordict as pa
from pandera import Check

# Define a schema
rl_schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(32, 64, 64, 3)),
        "action": pa.Tensor(dtype=torch.int64, shape=(32,), checks=Check.isin([0, 1, 2, 3])),
        "reward": pa.Tensor(dtype=torch.float32, shape=(32,)),
        "done": pa.Tensor(dtype=torch.bool, shape=(32,)),
    },
    batch_size=(32,)
)

# Validate a TensorDict
td = TensorDict({
    "observation": torch.randn(32, 64, 64, 3),
    "action": torch.randint(0, 4, (32,)),
    "reward": torch.randn(32),
    "done": torch.zeros(32, dtype=torch.bool)
}, batch_size=[32])
rl_schema.validate(td)  # Works!

# Validate a tensorclass (same schema!)
@tensorclass
class RLData:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

tc = RLData(
    observation=torch.randn(32, 64, 64, 3),
    action=torch.randint(0, 4, (32,)),
    reward=torch.randn(32),
    done=torch.zeros(32, dtype=torch.bool),
    batch_size=[32]
)
rl_schema.validate(tc)  # Works with tensorclass too!
```

#### Example 5: Coercion and Type Conversion

Pandera can automatically convert types when needed:

```python
import pandera.tensordict as pa
from pandera import Check

# Schema with coercion enabled
coercing_schema = pa.TensorDictSchema(
    keys={
        "features": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 128),
            checks=Check.in_range(0.0, 1.0)
        ),
        "labels": pa.Tensor(dtype=torch.int32, shape=(None,)),
    },
    batch_size=(100,),
    coerce=True  # Enable automatic dtype coercion
)

# Input has wrong dtype - will be coerced automatically
td = TensorDict({
    "features": torch.rand(100, 128),  # float64 by default
    "labels": torch.arange(100, dtype=torch.int64)
}, batch_size=[100])

validated = coercing_schema.validate(td)
# "features" is now float32, "labels" is now int32
```

#### Example 6: Validating Data at Rest (Serialization/Deserialization)

TensorDict supports efficient serialization to disk (using `memmap` for large datasets). When loading data back, validation ensures no corruption occurred:

```python
import torch
from tensordict import TensorDict, MemmapTensor
import pandera.tensordict as pa
from pandera import Check

# Schema for serializable data
dataset_schema = pa.TensorDictSchema(
    keys={
        "image": pa.Tensor(
            dtype=torch.uint8,
            shape=(None, 224, 224, 3),
        ),
        "label": pa.Tensor(dtype=torch.int64, shape=(None,)),
        "weight": pa.Tensor(
            dtype=torch.float32,
            shape=(None,),
            checks=Check.greater_than_or_equal_to(0.0),
        ),
        "metadata": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 10),
        ),
    },
    batch_size=(1000,),
)

# Save dataset to disk efficiently
def save_dataset(td: TensorDict, path: str):
    """Save TensorDict with memmap for large datasets."""
    td.memmap_(path)
    return td

# Load and validate - catches disk corruption, dtype mismatches, etc.
def load_and_validate_dataset(path: str) -> TensorDict:
    """Load dataset and validate before use."""
    td = TensorDict.load(path)
    
    # Validate schema compliance
    validated = dataset_schema.validate(td)
    
    return validated

# Create and save a large dataset (e.g., ImageNet subset)
dataset = TensorDict({
    "image": torch.randint(0, 256, (10000, 224, 224, 3), dtype=torch.uint8),
    "label": torch.randint(0, 1000, (10000,)),
    "weight": torch.rand(10000).abs(),
    "metadata": torch.randn(10000, 10),
}, batch_size=[10000])

save_dataset(dataset, "./data/image_dataset")

# Later: Load and validate before training
training_data = load_and_validate_dataset("./data/image_dataset")
# training_data is guaranteed to have correct dtypes, shapes, and value ranges
```

#### Example 7: Validating TensorClass Serialization

`tensorclass` objects can also be serialized. Pandera validates upon deserialization:

```python
import torch
from tensordict import TensorDict, tensorclass
import pandera.tensordict as pa
from pandera import Check

# Define a tensorclass for robot trajectory data
@tensorclass
class TrajectoryData:
    states: torch.Tensor      # (T, state_dim)
    actions: torch.Tensor    # (T, action_dim)
    rewards: torch.Tensor    # (T,)
    dones: torch.Tensor      # (T,)
    returns_to_go: torch.Tensor  # (T,)

# Schema for trajectory validation
trajectory_schema = pa.TensorDictSchema(
    keys={
        "states": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 64),  # state_dim
            checks=[
                Check.no_nan(),
                Check.no_inf(),
            ]
        ),
        "actions": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 8),  # action_dim
            checks=Check.in_range(-1.0, 1.0),  # Normalized actions
        ),
        "rewards": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "dones": pa.Tensor(dtype=torch.bool, shape=(None,)),
        "returns_to_go": pa.Tensor(
            dtype=torch.float32,
            shape=(None,),
            checks=Check.greater_than_or_equal_to(0.0),
        ),
    },
    batch_size=(None,)  # Variable length trajectories
)

# Save/Load tensorclass data
def save_trajectory(tc: TrajectoryData, path: str):
    """Serialize tensorclass to disk."""
    tc.save(path)

def load_trajectory(path: str) -> TrajectoryData:
    """Load and validate tensorclass from disk."""
    tc = TrajectoryData.load(path)
    
    # Validate - catches any corruption from disk I/O
    return trajectory_schema.validate(tc)

# Create and save trajectory
trajectory = TrajectoryData(
    states=torch.randn(100, 64),
    actions=torch.rand(100, 8) * 2 - 1,
    rewards=torch.randn(100),
    dones=torch.zeros(100, dtype=torch.bool),
    returns_to_go=torch.cumsum(torch.randn(100).abs(), dim=0),
    batch_size=[100]
)

save_trajectory(trajectory, "./data/trajectory.pt")

# Load before training - validates data integrity
loaded_trajectory = load_trajectory("./data/trajectory.pt")

# Now safe to use in offline RL algorithms (CQL, IQL, etc.)
```

#### Example 8: Batch Validation for Large-Scale Datasets

For large datasets, validate in batches to catch issues early:

```python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
from pandera import Check
from pathlib import Path

# Schema for a large stored dataset
large_dataset_schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 128)),
        "action": pa.Tensor(dtype=torch.float32, shape=(None, 16)),
        "reward": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "next_observation": pa.Tensor(dtype=torch.float32, shape=(None, 128)),
        "done": pa.Tensor(dtype=torch.bool, shape=(None,)),
    },
    batch_size=(1024,)
)

# Validate all batches in a directory of TensorDicts
def validate_dataset_directory(data_dir: Path) -> list[str]:
    """Validate all TensorDict files in a directory."""
    errors = []
    schema = large_dataset_schema
    
    for td_file in sorted(data_dir.glob("*.pt")):
        td = TensorDict.load(td_file)
        
        try:
            schema.validate(td, lazy=True)
        except pa.SchemaErrors as e:
            errors.append(f"{td_file}: {e}")
    
    return errors

# Check entire dataset before starting training
data_errors = validate_dataset_directory(Path("./data/experience_buffer"))

if data_errors:
    print("Dataset validation failed:")
    for err in data_errors:
        print(f"  - {err}")
else:
    print("Dataset validation passed - ready for training!")
```

#### Example 9: Validating TorchRL Rollouts

Real RL pipelines like those in [TorchRL](https://pytorch.org/rl/) rely on environment rollouts. Pandera ensures rollouts meet expected specifications before going into the replay buffer or training loop:

```python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
from pandera import Check

# Schema for a typical TorchRL environment rollout
rollout_schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 4, 64, 64),  # Image observation (CIFAR-like)
            checks=[
                Check.greater_than_or_equal_to(0.0),  # Pixel values normalized
                Check.less_than_or_equal_to(1.0),
            ]
        ),
        "action": pa.Tensor(dtype=torch.float32, shape=(None, 2)),
        "reward": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "done": pa.Tensor(dtype=torch.bool, shape=(None,)),
        "next": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 4, 64, 64),
        ),
    },
    batch_size=(256,)  # Rollout batch size
)

# Simulate a TorchRL environment rollout
def collect_rollout(env, policy, num_steps=256):
    rollout = env.rollout(max_steps=num_steps, policy=policy)
    return rollout

# Validate before adding to replay buffer
rollout = collect_rollout(env, policy)
validated_rollout = rollout_schema.validate(rollout)  # Fails fast if invalid

# Add to replay buffer - now we know data is valid!
replay_buffer.add(validated_rollout)
```

#### Example 7: Replay Buffer Validation with TorchRL

When sampling from replay buffers in large-scale training, you want to validate that the sampled batches meet your schema. This is especially important when using heterogeneous storage backends:

```python
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
import pandera.tensordict as pa
from pandera import Check

# Define schema for replay buffer contents
replay_schema = pa.TensorDictSchema(
    keys={
        "state": pa.Tensor(dtype=torch.float32, shape=(None, 64)),
        "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
        "reward": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "next_state": pa.Tensor(dtype=torch.float32, shape=(None, 64)),
        "done": pa.Tensor(dtype=torch.bool, shape=(None,)),
        "priority": pa.Tensor(dtype=torch.float32, shape=(None,)),
    },
    batch_size=(256,),
    # Allow dtype coercion for storage efficiency
    coerce=True,
)

# Create replay buffer (like in distributed training)
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=100000),
    batch_size=256,
)

# Validate sampled data before training
def train_step():
    batch = replay_buffer.sample()
    
    # Validate the batch - catch data corruption or storage errors
    validated_batch = replay_schema.validate(batch)
    
    # Now safe to pass to loss computation
    loss = compute_ppo_loss(validated_batch)
    loss.backward()
    optimizer.step()
```

#### Example 8: Actor-Critic Model Output Validation

In actor-critic methods (PPO, A2C, etc.), both the policy and value networks output structured data. Pandera validates these outputs match expected signatures:

```python
import torch
import pandera.tensordict as pa
from pandera import Check

# Schema for actor-critic network output
class ActorCriticOutput(pa.TensorDictModel):
    """Output from an actor-critic network."""
    
    action_mean: torch.Tensor = pa.Field(
        dtype=torch.float32,
        shape=(None, 4),  # 4D action space
        checks=Check.greater_than_or_equal_to(-2.0),
        Check.less_than_or_equal_to(2.0),
    )
    action_std: torch.Tensor = pa.Field(
        dtype=torch.float32,
        shape=(None, 4),
        checks=Check.greater_than(1e-6),  # Standard deviation > 0
    )
    value: torch.Tensor = pa.Field(
        dtype=torch.float32,
        shape=(None,),
        checks=Check.greater_than(-100.0),  # Reasonable value bounds
        Check.less_than(100.0),
    )
    
    class Config:
        batch_size = (32,)

# Validate model output before computing loss
model = ActorCriticNetwork()
batch = TensorDict({"observation": torch.randn(32, 64)}, batch_size=[32])

# Forward pass
output = model(batch)

# Validate output - ensures no NaN/Inf from unstable training
validated = ActorCriticOutput().validate(output)

# Safe to use in PPO loss
ppo_loss = compute_ppo_loss(validated["action_mean"], validated["action_std"], validated["value"])
```

#### Example 9: Distributed Training Data Validation

In distributed RL training, data flows between collectors, replay buffers, and trainers. Schema validation at each stage ensures data integrity:

```python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
from pandera import Check

# Define a unified schema for the entire training pipeline
training_schema = pa.TensorDictSchema(
    keys={
        # Environment interaction data
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 17)),
        "action": pa.Tensor(dtype=torch.float32, shape=(None, 6)),
        "reward": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "done": pa.Tensor(dtype=torch.bool, shape=(None,)),
        
        # PPO-specific data
        "log_prob": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "value": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "advantage": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "return": pa.Tensor(dtype=torch.float32, shape=(None,)),
    },
    batch_size=(512,),
)

# Validator for data collector (ensures valid env interactions)
collector_validator = training_schema

# Validator for trainer (additional PPO-specific checks)
trainer_validator = pa.TensorDictSchema(
    keys={
        **training_schema.keys,
        "log_prob": pa.Tensor(
            dtype=torch.float32,
            shape=(None,),
            checks=Check.less_than(0.0),  # Log prob should be negative
        ),
        "advantage": pa.Tensor(
            dtype=torch.float32,
            shape=(None,),
            checks=[
                Check.mean().greater_than(0.0),  # Positive advantage on average
            ],
        ),
    },
    batch_size=(512,),
)

# In your distributed training loop:
def distributed_train_step(collector_rank, trainer_rank):
    # Collect experiences
    experience = collector.collect_experiences()
    
    # Validate before sending to replay buffer
    validated_exp = collector_validator.validate(experience)
    
    # Send to buffer (RRef)
    replay_buffer_ref.add(validated_exp)
    
    # Trainer pulls batch
    batch = replay_buffer_ref.sample()
    
    # Validate before training
    validated_batch = trainer_validator.validate(batch)
    
    # Train
    optimizer.zero_grad()
    loss = compute_ppo_loss(validated_batch)
    loss.backward()
    optimizer.step()
```

#### Example 10: Integration with Custom TensorDictModules

For users building custom neural network modules with `tensordict.nn`, Pandera can validate inputs/outputs:

```python
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import pandera.tensordict as pa
from pandera import Check

# Custom module that processes observations
class CustomEncoder(TensorDictModule):
    def __init__(self):
        super().__init__(
            module=nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            ),
            in_keys=["observation"],
            out_keys=["encoded"],
        )

# Schema for encoder input/output
encoder_schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 128),
            checks=[
                Check.greater_than_or_equal_to(-10.0),  # Input bounds
                Check.less_than_or_equal_to(10.0),
                Check.no_inf(),  # No infinite values
                Check.no_nan(),  # No NaN values
            ]
        ),
        "encoded": pa.Tensor(
            dtype=torch.float32,
            shape=(None, 128),
            checks=[
                Check.no_nan(),
                Check.no_inf(),
            ]
        ),
    },
    batch_size=(32,)
)

# Validation hook for model inputs/outputs
def validate_encoder_io(td: TensorDict) -> TensorDict:
    """Validate encoder input/output for debugging."""
    return encoder_schema.validate(td)

# Usage in training loop
encoder = CustomEncoder()
batch = TensorDict({"observation": torch.randn(32, 128)}, batch_size=[32])

# Forward pass with validation
encoded = encoder(batch)  # First pass through model

# Validate - catches numerical instability
validated = validate_encoder_io(encoded)

print(f"Encoded mean: {validated['encoded'].mean():.4f}")
```

## 2. TensorDict Data Model Primer

Reference: [TensorDict documentation](https://pytorch.org/tensordict/)

### 2.1 Core Container Types

| TensorDict type | Description |
|---|---|
| `tensordict` | A dict-like container of tensors that share a common batch dimension. Analogous to a pandas `DataFrame`. |
| `tensorclass` | A class-based representation of a TensorDict where entries are mapped to class attributes. |

### 2.2 Key Concepts

- **batch_size** — A `Size` object representing the common leading dimensions of all tensors in the container. This is the fundamental structural constraint.
- **keys** — The names of the entries within the container.
- **tensors** — The actual PyTorch tensors stored within the container, all sharing the same `batch_size`.
- **dtype** — The expected data type for the underlying tensors.

---



## 3. Design Principles

### 3.1 Small Public API Surface

The public API should be minimal and consistent with other backends:
- **Schema classes**: `TensorDictSchema`.
- **Model classes**: `TensorDictModel`.
- **Component classes**: `Tensor` (analogous to `Column`).

### 3.2 Consistent with Existing Pandera Patterns

The implementation must follow pandera's layered architecture:
1. **API layer** (`pandera/api/tensordict/`)
2. **Backend layer** (`pandera/backends/tensordict/`)
3. **Engine layer** (`pandera/engines/tensordict_engine.py`)

### 3.3 Schema Components Mirror the pandas Pattern

- `TensorDictSchema` → standalone schema for a `tensordict` or `tensorclass`.
- `Tensor` → defines constraints for a single entry in the container (dtype, shape/size, checks).

### 3.4 Leverage `Check` for Data-Level Validation

Structural validation (batch size, keys, dtypes) is handled by schema keyword arguments. Data-level validation (tensor values, ranges, etc.) is handled via pandera's existing `Check` system.

---



## 4. Public API Design

### 4.1 Entry Point

```python
import pandera.tensordict as pa
```

### 4.2 `TensorDictSchema`

`TensorDictSchema` validates a `tensordict`.

```python
class TensorDictSchema(BaseSchema):
    def __init__(
        self,
        keys: dict[str, Tensor] | list[str] | None = None,
        batch_size: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        checks: Check | list[Check] | None = None,
        coerce: bool = False,
        nullable: bool = False,
        # ... other common params
    ):
        ...

    def validate(self, check_obj: TensorDict, ...) -> TensorDict:
        ...

    def validate(self, check_obj: TensorClass, ...) -> TensorClass:
        ...
```

### 4.3 `TensorDict` validation (for `tensorclass`)

Validates a `*tensorclass*` object, ensuring it adheres to the specified structure and types.

### 4.4 Class-Based Models (Declarative API)

Using `TensorDictModel` for class-based definitions.

```python
class MyTensorDictModel(pa.TensorDictModel):
    data: torch.Tensor = pa.Field(dtype=torch.float32, shape=(None, 10))
    label: torch.Tensor = pa.Field(dtype=torch.int64, shape=(None,))

    class Config:
        batch_size = (None,)
```

---

## 5. Implementation Roadmap

### Phase 1: Core Infrastructure & API
- Define `pandera/api/tensordict/` structure.
- Implement `TensorDictSchema` and `Tensor`.
- Implement basic batch size and key existence validation.
- Create the engine for PyTorch dtype resolution.

### Phase 2: Backend Implementation
- Implement `pandera/backends/tensordict/` logic.
- Implement structural checks (dtype, shape/size of entries).
- Integrate with Pandera's existing `Check` system for tensor value validation.
- Ensure support for both `TensorDict` and `tensorclass` objects.

### Phase 3: Advanced Features & Integration
- Support for `coerce=True` (dtype/shape coercion).
- Implementation of specialized TensorDict checks (e.g., cross-entry constraints).
- Comprehensive test suite coverage across all backend features.

---

## 6. Architecture

### 6.1 Module Layout

```
pandera/
├── api/
│   └── tensordict/
│       ├── __init__.py
│       ├── container.py        # TensorDictSchema
│       ├── components.py       # Tensor
│       ├── model.py               # TensorDictModel
│       └── types.py            # Type aliases
├── backends/
│   └── tensordict/
│       ├── __init__.py
│       ├── base.py             # TensorDictSchemaBackend
│       ├── checks.py           # TensorDictCheckBackend
│       └── register.py         # Backend registration
├── engines/
│   └── tensordict_engine.py    # Engine for torch dtype registry
├── typing/
│   └── tensordict.py           # Annotation types (TensorDict, TensorClass)
└── tensordict.py               # Entry point: import pandera.tensordict as pa

```
