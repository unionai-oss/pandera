# PyTorch TensorDict Integration Spec

> **Status:** Draft
> **Issue:** TBD
> **Author:** opencode
> **Related work:** [xarray-schema](https://github.com/xarray-contrib/xarray-schema)

---

## 1. Motivation

[TensorDict](https://pytorch.org/tensordict/) is a powerful container for managing groups of tensors with the same batch dimension. It is central to many modern PyTorch workflows, especially in reinforcement learning and large-scale distributed training.

Adding TensorDict support extends pandera's reach to the PyTorch ecosystem, providing a consistent validation API for high-dimensional tensor containers, alongside its existing tabular backends.

### Why Pandera?
While `tensordict` provides structural integrity (ensuring all tensors share a `batch_size`), it does not provide a declarative, schema-based way to validate the **data content** or **complex metadata constraints** without writing significant boilerplate. 

Pandera adds value by:
1.  **Declarative Data Validation:** Moving from imperative `assert` statements to structured, reusable schemas (`TensorDictSchema`, `TensorModel`).
2.  **Complex Value Checks:** Leveraging the existing Pandera `Check` ecosystem (e.g., `in_range`, `isin`, `regex`-like pattern matching on metadata) for tensor values and dimensions.
3.  **Unified API:** Providing the same validation experience as pandas/polars/xarray, allowing developers to use a single library for all data-centric pipelines.
4.  **Type Safety:** Integrating with Python type hints via `@check_types` to catch structural errors at development time.

### Usage Example: Adding Value to TensorDict
```python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

# The "Standard" way (Manual/Imperative)
td = TensorDict({"data": torch.randn(10, 5), "label": torch.tensor([1, 2])}, batch_size=[10])
assert td["data"].max() < 1.0  # Manual check - hard to scale and reuse
assert td["data"].dtype == torch.float32

# The Pandera way (Declarative/Reusable)
schema = pa.TensorDictSchema(
    keys={
        "data": pa.Tensor(dtype=torch.float32, shape=(None, 5), checks=pa.Check.in_range(-1, 1)),
        "label": pa.Tensor(dtype=torch.int64, shape=(None,))
    },
    batch_size=(None, 5)
)

validated_td = schema.validate(td) # Automatically performs all complex checks
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
- **Schema classes**: `TensorDictSchema`, `TensorClassSchema`.
- **Model classes**: `TensorDictModel`, `TensorClassModel`.
- **Component classes**: `Tensor` (analogous to `Column`).

### 3.2 Consistent with Existing Pandera Patterns

The implementation must follow pandera's layered architecture:
1. **API layer** (`pandera/api/tensordict/`)
2. **Backend layer** (`pandera/backends/tensordict/`)
3. **Engine layer** (`pandera/engines/tensordict_engine.py`)

### 3.3 Schema Components Mirror the pandas Pattern

- `TensorDictSchema` → standalone schema for a `tensordict`.
- `TensorClassSchema` → schema for a `tensorclass` object.
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

### 4.3 `TensorClassSchema` (for `tensorclass`)

Validates a `tensorclass` object, ensuring it adheres to the specified structure and types.

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
- Implement `TensorDictSchema` and `TensorDictEntry`.
- Implement basic batch size and key existence validation.
- Create the engine for PyTorch dtype resolution.

### Phase 2: Backend Implementation
- Implement `pandera/backends/tensordict/` logic.
- Implement structural checks (dtype, shape/size of entries).
- Integrate with Pandera's existing `Check` system for tensor value validation.

### Phase 3: TensorClass & Declarative API
- Implement `TensorClassSchema` and `TensorClassModel`.
- Enable support for `tensorclass` objects through the declarative API.
- Ensure `@check_types` works with `TensorDict` annotations.

### Phase 4: Advanced Features & Integration
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
│       ├── container.py        # TensorDictSchema, TensorClassSchema
│       ├── components.py       # Tensor
│       ├── model.py               # TensorDictModel, TensorClassModel
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
