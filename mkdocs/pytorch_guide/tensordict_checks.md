# Checks and Lazy Validation

Use [`Check`](../reference/generated/pandera.api.checks.Check.html) to validate tensor values.

## Apply checks to Tensor components

```python
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
    {"values": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])},
    batch_size=[10],
)
validated = schema.validate(td)
```

## Lazy validation

Set `lazy=True` to collect all errors:

```python
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

## Failure cases point at the failing elements

When a value check fails, the reported error carries the exact tensor
indices that violated the check, so you can locate the offending elements
without searching the tensor by hand:

```python
failure_schema = pa.TensorDictSchema(
    keys={
        "values": pa.Tensor(
            dtype=torch.float32,
            checks=[Check.greater_than(0.0)],
        ),
    },
)

td_fail = TensorDict(
    {"values": torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])},
    batch_size=[5],
)

try:
    failure_schema.validate(td_fail)
except pa.errors.SchemaError as exc:
    print(exc)
    # The failure cases include the violating indices and values
    print("indices:", exc.failure_cases["index"])
    print("values:", exc.failure_cases["failure_case"])
```

For N-D tensors the indices are coordinate tuples; for batch-level checks
(reduced over trailing dims, e.g. a one-hot row check) the reported index
is the batch index of the failing row.

## Available checks

Standard pandera checks work with tensor data:

- [`greater_than`](../reference/generated/pandera.api.checks.Check.html)
- [`less_than`](../reference/generated/pandera.api.checks.Check.html)
- [`greater_than_or_equal_to`](../reference/generated/pandera.api.checks.Check.html)
- [`less_than_or_equal_to`](../reference/generated/pandera.api.checks.Check.html)
- [`in_range`](../reference/generated/pandera.api.checks.Check.html)
- [`isin`](../reference/generated/pandera.api.checks.Check.html)
- [`notin`](../reference/generated/pandera.api.checks.Check.html)
- [`str_matches`](../reference/generated/pandera.api.checks.Check.html)
- [`str_contains`](../reference/generated/pandera.api.checks.Check.html)
- [`str_startswith`](../reference/generated/pandera.api.checks.Check.html)
- [`str_endswith`](../reference/generated/pandera.api.checks.Check.html)

## See also

- [Checks](../checks.html) — full Check API reference
- [Error Reporting](./error_reporting.html) — error handling
