"""Parser backend for pandas"""

from functools import partial
from typing import Optional, Union

import pandas as pd

from pandera.api.base.parsers import ParserResult
from pandera.api.pandas.types import is_field, is_table
from pandera.api.parsers import Parser
from pandera.backends.base import BaseParserBackend
from pandera.engines.pandas_engine import PANDAS_3_0_0_PLUS


class PandasParserBackend(BaseParserBackend):
    """Parser backend of pandas."""

    def __init__(self, parser: Parser):
        """Initializes a parser backend object."""
        super().__init__(parser)
        assert parser._parser_fn is not None, "Parser._parser_fn must be set."
        self.parser = parser
        self.parser_fn = partial(parser._parser_fn, **parser._parser_kwargs)

    def preprocess(
        self, parse_obj, key
    ) -> pd.Series | pd.DataFrame | dict[str, pd.DataFrame]:
        """Preprocesses a parser object before applying the parse function."""
        if is_table(parse_obj) and key is not None:
            return self.preprocess_table_with_key(parse_obj, key)
        elif is_table(parse_obj) and key is None:
            return self.preprocess_table(parse_obj)
        else:
            return parse_obj

    def preprocess_table_with_key(
        self,
        parse_obj,
        key,
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        return parse_obj[key]

    def preprocess_table(
        self, parse_obj
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        return parse_obj

    def apply(self, parse_obj):
        """Apply the parse function to a parser object."""
        if is_field(parse_obj):
            return self.apply_field(parse_obj)
        elif is_table(parse_obj):
            return self.apply_table(parse_obj)
        else:
            raise NotImplementedError

    def apply_field(self, parse_obj):
        if self.parser.element_wise:
            return parse_obj.map(self.parser_fn)
        return self.parser_fn(parse_obj)

    def apply_table(self, parse_obj):
        if self.parser.element_wise:
            # pandas 3.0 removed applymap, use map instead
            if PANDAS_3_0_0_PLUS:
                return parse_obj.map(self.parser_fn)
            else:
                return parse_obj.applymap(self.parser_fn)  # type: ignore[attr-defined]
        return self.parser_fn(parse_obj)

    def postprocess(
        self,
        parse_obj,
        parser_output,
    ) -> ParserResult:
        """Postprocesses the result of applying the parser function."""
        return ParserResult(
            parser_output=parser_output, parsed_object=parse_obj
        )

    def __call__(
        self,
        parse_obj: Union[pd.Series, pd.DataFrame],
        key: str | None = None,
    ):
        parse_obj = self.preprocess(parse_obj, key)
        parser_output = self.apply(parse_obj)
        return self.postprocess(parse_obj, parser_output)
