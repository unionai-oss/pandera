"""Pandera type annotations for TensorDict."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class TensorDictAnnotationBase:
    """Marker base for pandera TensorDict annotation types.

    Used by :attr:`~pandera.typing.common.AnnotationInfo.is_generic_tensordict`
    to recognise ``TensorDict[Model]`` annotations in decorator-based
    validation (e.g. :func:`~pandera.decorators.check_types`).
    """

    __slots__ = ()


class TensorDict(TensorDictAnnotationBase, Generic[T]):
    """Annotation-only generic for validating :class:`tensordict.TensorDict`.

    Usage::

        import pandera.tensordict as pa
        from pandera.typing.tensordict import TensorDict
        from pandera import check_types

        class Observation(pa.TensorDictModel):
            next_obs: torch.float32 = pa.Field(...)
            ...

        @check_types
        def fn(td: TensorDict[Observation]) -> TensorDict[Observation]:
            ...
    """

    __slots__ = ()


__all__ = [
    "TensorDict",
    "TensorDictAnnotationBase",
    "T",
]
