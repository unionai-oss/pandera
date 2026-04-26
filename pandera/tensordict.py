"""Pandera TensorDict API."""

from __future__ import annotations

import warnings

from pandera import errors

try:
    from pandera.engines.tensordict_engine import DataType as _DataType

    DataType: type[_DataType] | None = _DataType
except ImportError:
    DataType = None

from pandera.api.tensordict.components import Tensor
from pandera.api.tensordict.container import TensorDictSchema
from pandera.api.tensordict.model import TensorDictModel
from pandera.api.tensordict.model_components import Field

__all__ = [
    "Tensor",
    "TensorDictSchema",
    "TensorDictModel",
    "Field",
    "errors",
    "DataType",
    "SchemaError",
    "SchemaErrors",
]

try:
    from pandera.schema_inference.tensordict import infer_schema

    __all__.append("infer_schema")
except ImportError as e:
    warnings.warn(f"Could not import infer_schema: {e}")
    infer_schema = None

from pandera.errors import SchemaError, SchemaErrors

# Import IO module if available
try:
    from pandera.io.tensordict_io import (
        from_json,
        from_yaml,
        load,
        save,
        to_json,
        to_yaml,
    )

    __all__.extend(
        [
            "from_json",
            "from_yaml",
            "to_json",
            "to_yaml",
            "save",
            "load",
        ]
    )
except ImportError:
    pass
