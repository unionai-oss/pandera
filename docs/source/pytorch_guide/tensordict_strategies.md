---
file_format: mystnb
---

(pytorch-tensordict-strategies)=

# Hypothesis Strategies

Generate synthetic TensorDict data for property-based testing using
[Hypothesis](https://hypothesis.readthedocs.io/).

## Installation

```bash
pip install pandera[strategies]
```

## Basic Usage

Generate valid TensorDicts from a schema:

```{code-cell} python
from hypothesis import given, settings
import torch
from tensordict import TensorDict
import pandera.tensordict as pa
from pandera.strategies import tensordict_strategy

# Define schema
schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
        "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
    },
    batch_size=(32,),
)

# Generate and validate TensorDicts
@given(tensordict_strategy(schema))
@settings(max_examples=10)
def test_policy_output(td):
    """Property-based test for policy network."""
    # Schema validates automatically
    assert td["observation"].shape == (32, 10)
    assert td["action"].dtype == torch.int64

# Run the test
test_policy_output()
```

## TensorClass Generation

Generate tensorclass instances:

```{code-cell} python
from tensordict import tensorclass
from pandera.strategies import tensorclass_strategy

@tensorclass
class RLData:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 64)),
        "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
        "reward": pa.Tensor(dtype=torch.float32, shape=(None,)),
    },
    batch_size=(128,),
)

@given(tensorclass_strategy(RLData, schema))
@settings(max_examples=10)
def test_tensorclass(tc):
    assert tc.batch_size == [128]
    assert hasattr(tc, "observation")

test_tensorclass()
```

## Custom Data Generation

Combine with Hypothesis strategies for more control:

```{code-cell} python
from hypothesis import given, settings
import torch
import pandera.tensordict as pa
from pandera.strategies import tensordict_strategy
from pandera import Check

# Generate only valid actions (0-3)
schema = pa.TensorDictSchema(
    keys={
        "action": pa.Tensor(dtype=torch.int64, shape=(None,), checks=Check.isin([0, 1, 2, 3])),
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

@given(tensordict_strategy(schema))
@settings(max_examples=10)
def test_action_space(td):
    """Test that actions are always in valid range."""
    assert td["action"].max() < 4
    assert td["action"].min() >= 0

test_action_space()
```

## Integration with PyTorch Testing Patterns

```{code-cell} python
import torch
from hypothesis import given, settings
import pandera.tensordict as pa
from pandera.strategies import tensordict_strategy

class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs)

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

@given(tensordict_strategy(schema))
@settings(max_examples=10)
def test_policy_model(td):
    model = PolicyNetwork()
    
    # Test with various inputs
    output = model(td["observation"])
    assert output.shape == (32, 5)
    assert torch.isfinite(output).all()

test_policy_model()
```

## Use Cases

### 1. Unit Testing Models

Generate diverse inputs for model testing:

```{code-cell} python
import torch
from hypothesis import given, settings
import pandera.tensordict as pa
from pandera.strategies import tensordict_strategy

class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs)

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

@given(tensordict_strategy(schema))
@settings(max_examples=50)
def test_policy_model(td):
    model = PolicyNetwork()
    
    # Test with various inputs
    output = model(td["observation"])
    assert output.shape == (32, 5)
    assert torch.isfinite(output).all()

test_policy_model()
```

### 2. Property-Based Testing of RL Algorithms

Test reinforcement learning algorithms with generated rollouts:

```{code-cell} python
from tensordict import tensorclass
from hypothesis import given, settings
import pandera.tensordict as pa
from pandera.strategies import tensorclass_strategy

@tensorclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

rollout_schema = pa.TensorDictSchema(
    keys={
        "observations": pa.Tensor(dtype=torch.float32, shape=(None, 17)),
        "actions": pa.Tensor(dtype=torch.float32, shape=(None, 6)),
        "rewards": pa.Tensor(dtype=torch.float32, shape=(None,)),
        "dones": pa.Tensor(dtype=torch.bool, shape=(None,)),
    },
    batch_size=(256,),
)

class ReplayBuffer:
    def __init__(self):
        self.data = []
    
    def add(self, rollout: Rollout):
        self.data.append(rollout)
    
    def sample(self) -> Rollout:
        import random
        return random.choice(self.data) if self.data else None

@given(tensorclass_strategy(Rollout, rollout_schema))
@settings(max_examples=10)
def test_replay_buffer(rollout):
    """Test replay buffer with generated rollouts."""
    replay_buffer = ReplayBuffer()
    replay_buffer.add(rollout)
    sampled = replay_buffer.sample()
    
    # Properties
    assert sampled is not None
    assert "observations" in sampled.keys()

test_replay_buffer()
```

### 3. Stress Testing Data Preprocessing

Test preprocessing pipelines with edge cases:

```{code-cell} python
import torch
from hypothesis import given, settings
import pandera.tensordict as pa
from pandera.strategies import tensordict_strategy

def preprocess(td: TensorDict) -> TensorDict:
    """Normalize observations."""
    td["observation"] = (td["observation"] - td["observation"].mean()) / td["observation"].std()
    return td

schema = pa.TensorDictSchema(
    keys={
        "observation": pa.Tensor(dtype=torch.float32, shape=(None, 10)),
    },
    batch_size=(32,),
)

@given(tensordict_strategy(schema))
@settings(max_examples=50)
def test_preprocessing(td):
    """Test preprocessing handles various inputs."""
    processed = preprocess(td)
    
    # Check normalization
    assert torch.allclose(processed["observation"].mean(), torch.tensor(0.0), atol=1e-4)

test_preprocessing()
```

## See also

- {ref}`pytorch-tensordict-schema` — define validation schemas
- {ref}`pytorch-checks` — value constraints
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
