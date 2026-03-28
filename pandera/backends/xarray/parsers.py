"""Parser backend for xarray."""

from __future__ import annotations

from functools import partial

from pandera.api.base.parsers import ParserResult
from pandera.api.parsers import Parser
from pandera.backends.base import BaseParserBackend


class XarrayParserBackend(BaseParserBackend):
    """Apply :class:`~pandera.api.parsers.Parser` to xarray objects."""

    def __init__(self, parser: Parser):
        super().__init__(parser)
        assert parser._parser_fn is not None, "Parser._parser_fn must be set."
        self.parser = parser
        self.parser_fn = partial(parser._parser_fn, **parser._parser_kwargs)

    def preprocess(self, parse_obj, key: str | None):
        import xarray as xr

        if key is None:
            return parse_obj
        if isinstance(parse_obj, xr.Dataset):
            return parse_obj[key]
        raise TypeError(
            f"cannot select key {key!r} on {type(parse_obj).__name__}"
        )

    def apply(self, parse_obj):
        if self.parser.element_wise:
            return self._apply_elementwise(parse_obj)
        return self.parser_fn(parse_obj)

    def _apply_elementwise(self, parse_obj):
        import numpy as np
        import xarray as xr

        if not isinstance(parse_obj, xr.DataArray):
            raise NotImplementedError(
                "element-wise parsers on non-DataArray xarray objects."
            )

        def _fn(x):
            return self.parser_fn(x)

        vectorized = np.vectorize(_fn, otypes=[object])
        out = vectorized(parse_obj.values)
        return xr.DataArray(out, dims=parse_obj.dims, coords=parse_obj.coords)

    def postprocess(self, parse_obj, parser_output) -> ParserResult:
        return ParserResult(
            parser_output=parser_output,
            parsed_object=parse_obj,
        )

    def __call__(self, parse_obj, key: str | None = None) -> ParserResult:
        prepped = self.preprocess(parse_obj, key)
        applied = self.apply(prepped)
        return self.postprocess(prepped, applied)
