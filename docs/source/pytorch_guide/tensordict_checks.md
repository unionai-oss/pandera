---
file_format: mystnb
---

(pytorch-checks)=

# Checks and Lazy Validation

Use {class}`~pandera.api.checks.Check` to validate tensor values.

## Apply checks to Tensor components

```{code-cell} python
import torch
from tensordict import TensorDict
from pandera import Check
import pandera.tensordict as pa

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

td = TensorDict(
    {"values": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])},
    batch_size=[10],
)
validated = schema.validate(td)
```

## Lazy validation

Set `lazy=True` to collect all errors:

```{code-cell} python
td_invalid = TensorDict(
    {"values": torch.tensor([-0.1, 0.2, 0.3, 10.0, 0.5])},
    batch_size=[5],
)

try:
    schema.validate(td_invalid, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(f"Found {len(exc.schema_errors)} errors")
    print(exc)
```

## Available checks

Standard pandera checks work with tensor data:

- {func}`~pandera.api.checks.Check.greater_than`
- {func}`~pandera.api.checks.Check.less_than`
- {func}`~pandera.api.checks.Check.greater_than_or_equal_to`
- {func}`~pandera.api.checks.Check.less_than_or_equal_to`
- {func}`~pandera.api.checks.Check.in_range`
- {func}`~pandera.api.checks.Check.isin`
- {func}`~pandera.api.checks.Check.notin`
- {func}`~pandera.api.checks.Check.str_matches`
- {func}`~pandera.api.checks.Check.str_contains`
- {func}`~pandera.api.checks.Check.str_startswith`
- {func}`~pandera.api.checks.Check.str_endswith`

## See also

- {ref}`checks` — full Check API reference
- {ref}`pytorch-error-reporting` — error handling
