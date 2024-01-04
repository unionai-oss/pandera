"""Ibis engine and data types."""

import dataclasses
from typing import Any, Iterable, Union

import ibis
import ibis.expr.datatypes as dt
import numpy as np

from pandera import dtypes
from pandera.dtypes import immutable
from pandera.engines import engine, numpy_engine


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Ibis data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native Ibis dtype boxed by the data type."""

    def __init__(self, dtype: Any):
        super().__init__()
        object.__setattr__(self, "type", ibis.dtype(dtype))
        dtype_cls = dtype if inspect.isclass(dtype) else dtype.__class__
        warnings.warn(
            f"'{dtype_cls}' support is not guaranteed.\n"
            + "Usage Tip: Consider writing a custom "
            + "pandera.dtypes.DataType or opening an issue at "
            + "https://github.com/pandera-dev/pandera"
        )

    def __post_init__(self):
        # this method isn't called if __init__ is defined
        object.__setattr__(self, "type", ibis.dtype(self.type))  # pragma: no cover

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Any,
    ) -> Union[bool, Iterable[bool]]:
        try:
            return self.type == pandera_dtype.type
        except TypeError:
            return False


class Engine(
    metaclass=engine.Engine,
    base_pandera_dtypes=(DataType, numpy_engine.DataType),
):
    """Ibis data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a pandas-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            # TODO(deepyaman): Replace below with `data_type.to_numpy()`
            #   once have https://github.com/ibis-project/ibis/pull/7910
            from ibis.formats.numpy import NumpyType

            np_dtype = NumpyFormat.from_dtype(data_type)

        return engine.Engine.dtype(cls, np_dtype)


###############################################################################
# float
###############################################################################


@Engine.register_dtype(
    equivalents=[np.float64, dtypes.Float64, dtypes.Float64(), dt.Float64, dt.float64]
)
@immutable
class Float64(DataType, dtypes.Float64):
    """Semantic representation of a :class:`dt.Float64`."""

    type = dt.float64
