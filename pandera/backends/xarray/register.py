"""Register xarray backends."""

from functools import lru_cache
from typing import Optional

from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser
from pandera.api.xarray.container import DataArraySchema, DatasetSchema
from pandera.backends.xarray.checks import XarrayCheckBackend
from pandera.backends.xarray.container import (
    DataArraySchemaBackend,
    DatasetSchemaBackend,
)
from pandera.backends.xarray.hypotheses import XarrayHypothesisBackend
from pandera.backends.xarray.parsers import XarrayParserBackend


@lru_cache
def register_xarray_backends(
    check_cls_fqn: str | None = None,
):
    """Register xarray backends.

    This function is called at schema initialization in the register_default_backends
    method.

    :param check_cls_fqn: name of the check class to register backends for.
    """
    import pandera.backends.xarray.builtin_checks  # noqa: F401
    from pandera.api.xarray.types import XARRAY_CHECK_OBJECT_TYPES

    for obj_type in XARRAY_CHECK_OBJECT_TYPES:
        try:
            import xarray as xr

            if obj_type is xr.DataArray:
                Check.register_backend(obj_type, XarrayCheckBackend)
                Hypothesis.register_backend(obj_type, XarrayHypothesisBackend)
                Parser.register_backend(obj_type, XarrayParserBackend)
                DataArraySchema.register_backend(
                    obj_type, DataArraySchemaBackend
                )
            elif obj_type is xr.Dataset:
                Check.register_backend(obj_type, XarrayCheckBackend)
                Hypothesis.register_backend(obj_type, XarrayHypothesisBackend)
                Parser.register_backend(obj_type, XarrayParserBackend)
                DatasetSchema.register_backend(obj_type, DatasetSchemaBackend)
        except ImportError:
            pass
