---
file_format: mystnb
---

(pytorch-error-reporting)=

# Error Reporting

Pandera raises {class}`~pandera.errors.SchemaError` for individual validation failures
and {class}`~pandera.errors.SchemaErrors` when `lazy=True`.

## Non-lazy validation (fail-fast)

By default, validation stops on the first error:

```{code-cell} python
import torch
from tensordict import TensorDict
import pandera.tensordict as pa

schema = pa.TensorDictSchema(
    keys={"x": pa.Tensor(dtype=torch.float32, shape=(None, 10))},
    batch_size=(32,),
)

td = TensorDict({"x": torch.randn(16, 10)}, batch_size=[16])

try:
    schema.validate(td)
except pa.errors.SchemaError as exc:
    print(f"Error: {exc}")
    print(f"Reason: {exc.reason_code}")
```

## Lazy validation

Set `lazy=True` to collect all errors:

```{code-cell} python
td_invalid = TensorDict(
    {"x": torch.randn(16, 10), "y": torch.randn(16, 10)},
    batch_size=[16],
)

try:
    schema.validate(td_invalid, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(f"Found {len(exc.schema_errors)} errors")
    for err in exc.schema_errors:
        print(f"  - {err.message}")
    print(f"\nError counts: {exc.error_counts}")
```

## Error reason codes

- `WRONG_DATATYPE` — tensor dtype mismatch
- `COLUMN_NOT_IN_DATAFRAME` — missing key
- `CHECK_ERROR` — value check failure

## See also

- {ref}`checks` — Check API
- {ref}`pytorch-tensordict-schema` — TensorDictSchema
- {ref}`lazy-validation` — general lazy validation guide