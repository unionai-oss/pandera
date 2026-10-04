"""Serialize and deserialize TensorDict schemas."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pandera.api.checks import Check
from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.container import TensorDictSchema
from pandera.engines import tensordict_engine
from pandera.errors import SchemaDefinitionError
from pandera.io._check_io import checks_dict_to_list


def serialize_schema(
    dataframe_schema, *, minimal: bool = True
) -> dict[str, Any]:
    """Serialize a TensorDict schema into JSON/YAML-compatible dict."""
    from pandera import __version__

    statistics: dict[str, Any] = {
        "keys": {},
        "batch_size": (
            list(dataframe_schema.batch_size)
            if dataframe_schema.batch_size
            else None
        ),
    }

    if dataframe_schema.keys:
        for key_name, tensor_component in dataframe_schema.keys.items():
            statistics["keys"][key_name] = {
                "dtype": (
                    str(tensor_component.dtype)
                    if tensor_component.dtype is not None
                    else None
                ),
                "shape": list(tensor_component.shape)
                if tensor_component.shape
                else None,
                "nullable": getattr(tensor_component, "nullable", False),
                "checks": (
                    [
                        {
                            "options": {"check_name": check.name},
                            **(check.statistics or {}),
                        }
                        for check in tensor_component.checks
                    ]
                    if hasattr(tensor_component, "checks")
                    and tensor_component.checks
                    else None
                ),
            }

    out = {
        "schema_type": "tensordict",
        "version": __version__,
        "keys": statistics["keys"],
        "batch_size": statistics["batch_size"],
        "coerce": dataframe_schema.coerce,
        "dtype": (
            str(dataframe_schema.dtype) if dataframe_schema.dtype else None
        ),
    }

    return out


def _deserialize_check_stats(check, serialized_check_stats):
    """Deserialize check statistics and reconstruct Check with options."""
    options = {}
    if isinstance(serialized_check_stats, dict):
        options = serialized_check_stats.pop("options", {})

        if (
            "value" in serialized_check_stats
            and len(serialized_check_stats) == 1
        ):
            serialized_check_stats = serialized_check_stats["value"]

    check_instance = check(**serialized_check_stats)

    if options:
        for option_name, option_value in options.items():
            if option_name != "check_name":
                setattr(check_instance, option_name, option_value)

    return check_instance


def _deserialize_tensor_stats(serialized_tensor_stats):
    """Deserialize Tensor component stats and reconstruct Tensor."""
    serialized_tensor_stats = dict(serialized_tensor_stats)

    dtype = serialized_tensor_stats.get("dtype")
    if dtype:
        try:
            dtype = tensordict_engine.Engine.dtype(dtype)
        except (TypeError, ValueError):
            dtype = tensordict_engine.Engine.dtype(str(dtype).lower())

    shape = serialized_tensor_stats.get("shape")
    if shape is not None:
        shape = tuple(shape)

    checks = checks_dict_to_list(serialized_tensor_stats.get("checks"))
    if checks is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check["options"]["check_name"]), check
            )
            for check in checks
        ]

    return {
        "dtype": dtype,
        "shape": shape,
        "nullable": serialized_tensor_stats.get("nullable", False),
        "checks": checks,
        **{
            key: serialized_tensor_stats.get(key)
            for key in ["description", "title"]
            if key in serialized_tensor_stats
        },
    }


def deserialize_schema(serialized_schema):
    """Deserialize a mapping into TensorDictSchema."""
    serialized_schema = serialized_schema if serialized_schema else {}

    if not isinstance(serialized_schema, Mapping):
        raise SchemaDefinitionError("Schema representation must be a mapping.")

    st = serialized_schema.get("schema_type")
    if st is not None and st != "tensordict":
        raise SchemaDefinitionError(
            f"Expected schema_type 'tensordict', got {st!r}"
        )

    keys = serialized_schema.get("keys")
    if keys is not None:
        keys = {
            key_name: Tensor(**_deserialize_tensor_stats(tensor_stats))
            for key_name, tensor_stats in keys.items()
        }

    batch_size = serialized_schema.get("batch_size")
    if batch_size is not None:
        batch_size = tuple(batch_size)

    return TensorDictSchema(
        keys=keys,
        batch_size=batch_size,
        coerce=serialized_schema.get("coerce", False),
        dtype=serialized_schema.get("dtype"),
    )


def from_yaml(yaml_schema):
    """Load a TensorDictSchema from YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for YAML support. "
            "Install with: pip install pyyaml"
        ) from exc

    try:
        with Path(yaml_schema).open("r", encoding="utf-8") as f:
            serialized_schema = yaml.safe_load(f)
    except (TypeError, OSError):
        serialized_schema = yaml.safe_load(yaml_schema)

    return deserialize_schema(serialized_schema)


def to_yaml(dataframe_schema, stream=None, *, minimal: bool = True):
    """Write a TensorDictSchema to YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for YAML support. "
            "Install with: pip install pyyaml"
        ) from exc

    statistics = serialize_schema(dataframe_schema, minimal=minimal)

    def _write_yaml(obj, stream):
        return yaml.safe_dump(obj, stream=stream, sort_keys=False)

    try:
        with Path(stream).open("w", encoding="utf-8") as f:
            _write_yaml(statistics, f)
    except (TypeError, OSError):
        return _write_yaml(statistics, stream)


def from_json(source):
    """Load a TensorDictSchema from JSON."""
    if isinstance(source, str):
        try:
            serialized_schema = json.loads(source)
        except json.decoder.JSONDecodeError:
            with Path(source).open(encoding="utf-8") as f:
                serialized_schema = json.load(fp=f)
    elif isinstance(source, Path):
        with source.open(encoding="utf-8") as f:
            serialized_schema = json.load(fp=f)
    else:
        serialized_schema = json.load(fp=source)

    return deserialize_schema(serialized_schema)


def to_json(dataframe_schema, target=None, *, minimal: bool = True, **kwargs):
    """Write a TensorDictSchema to JSON."""
    serialized_schema = serialize_schema(dataframe_schema, minimal=minimal)

    if target is None:
        return json.dumps(serialized_schema, sort_keys=False, **kwargs)

    if isinstance(target, (str, Path)):
        with Path(target).open("w", encoding="utf-8") as f:
            json.dump(serialized_schema, fp=f, sort_keys=False, **kwargs)
    else:
        json.dump(serialized_schema, fp=target, sort_keys=False, **kwargs)


def save(dataframe_schema, tensor_dict, path):
    """Save TensorDict to disk with schema metadata embedded."""
    import torch

    path = Path(path)

    if isinstance(tensor_dict, dict):
        from tensordict import TensorDict

        td = TensorDict(tensor_dict)
        # Infer batch dimensions from the tensors so that the saved
        # TensorDict round-trips through schema validation in ``load``.
        batch_dims = (
            len(dataframe_schema.batch_size)
            if dataframe_schema.batch_size is not None
            else None
        )
        td.auto_batch_size_(batch_dims=batch_dims)
    else:
        td = tensor_dict

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "tensordict": td,
            "schema": serialize_schema(dataframe_schema),
        },
        path,
    )


def load(path):
    """Load TensorDict from disk and validate with schema."""
    import torch

    data = torch.load(path, weights_only=False)

    if isinstance(data, dict) and "tensordict" in data:
        td = data["tensordict"]
        saved_schema = data.get("schema")

        if saved_schema:
            loaded_schema = deserialize_schema(saved_schema)
            return loaded_schema.validate(td)

    if isinstance(data, dict):
        try:
            from tensordict import TensorDict

            from pandera.schema_inference.tensordict import infer_schema

            if any(hasattr(v, "dtype") for v in data.values()):
                td = TensorDict(data)

                try:
                    schema = infer_schema(td)
                    return schema.validate(td)
                except Exception:
                    return td
        except ImportError:
            pass

    return data
