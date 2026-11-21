# Error Reporting

Pandera raises [`SchemaError`](../reference/generated/pandera.errors.SchemaError.html) for individual validation failures
and [`SchemaErrors`](../reference/generated/pandera.errors.SchemaErrors.html) when `lazy=True`.

## Non-lazy validation (fail-fast)

By default, validation stops on the first error:

```python
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

```python
td_invalid = TensorDict(
    {"x": torch.randn(16, 10), "y": torch.randn(16, 10)},
    batch_size=[16],
)

try:
    schema.validate(td_invalid, lazy=True)
except pa.errors.SchemaErrors as exc:
    print(f"Found {len(exc.schema_errors)} errors")
    for err in exc.schema_errors:
        print(f"  - {str(err)}")
    print(f"\nError counts: {exc.error_counts}")
```

## Error reason codes

- `WRONG_DATATYPE` — tensor dtype mismatch
- `COLUMN_NOT_IN_DATAFRAME` — missing key
- `CHECK_ERROR` — value check failure

## See also

- [Checks](../checks.html) — Check API
- [TensorDictSchema](./tensordict_schema.html) — TensorDictSchema
- [Lazy Validation](../lazy_validation.html) — general lazy validation guide
