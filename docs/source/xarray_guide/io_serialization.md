---
file_format: mystnb
---

(xarray-io-serialization)=

# IO Serialization

Schemas can be serialized to **YAML** or **JSON** for storage, version
control, or sharing.

## YAML

```{code-cell} python
import numpy as np
import pandera.xarray as pa

schema = pa.DataArraySchema(
    dtype=np.float64,
    dims=("x", "y"),
    name="temperature",
    nullable=False,
    coerce=True,
)

yaml_str = pa.to_yaml(schema)
print(yaml_str)
```

Restore the schema from YAML:

```{code-cell} python
restored = pa.from_yaml(yaml_str)
print(type(restored).__name__, restored.name)
```

## JSON

```{code-cell} python
json_str = pa.to_json(schema)
restored = pa.from_json(json_str)
print(type(restored).__name__, restored.name)
```

Both formats support `DatasetSchema` as well:

```{code-cell} python
ds_schema = pa.DatasetSchema(
    data_vars={
        "temp": pa.DataVar(dtype="float64", dims=("x",)),
    },
    coords={"x": pa.Coordinate(dtype="float64")},
)
json_str = pa.to_json(ds_schema)
restored = pa.from_json(json_str)
print(type(restored).__name__)
print(list(restored.data_vars.keys()))
```

## File round-trip

Write to and read from files:

```python
pa.to_yaml(schema, stream="schema.yaml")
schema = pa.from_yaml("schema.yaml")

pa.to_json(schema, target="schema.json")
schema = pa.from_json("schema.json")
```

```{admonition} See also
:class: tip

{ref}`xarray-schema-inference` for automatically inferring schemas from data.
```
