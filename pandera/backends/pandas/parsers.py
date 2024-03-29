"Parser backend for pandas"

from functools import partial
from typing import Dict, Optional, Union

import pandas as pd
from multimethod import DispatchError, overload

from pandera.api.base.parsers import ParserResult
from pandera.api.parsers import Parser
from pandera.backends.base import BaseParserBackend
from pandera.api.pandas.types import (
    is_field,
    is_table,
)


class PandasParserBackend(BaseParserBackend):
    """Parser backend of pandas."""

    def __init__(self, parser: Parser):
        """Initializes a parser backend object."""
        super().__init__(parser)
        assert parser._parser_fn is not None, "Parser._parser_fn must be set."
        self.parser = parser
        self.parser_fn = partial(parser._parser_fn, **parser._parser_kwargs)

    @overload
    def prerprocess(
        self, parse_obj, key
    ) -> pd.Series:  # pylint:disable=unused-argument
        """Preprocesses a parser object before applying the parse function."""
        return parse_obj

    @overload  # type: ignore [no-redef]
    def prerprocess(
        self,
        parse_obj: is_table,  # type: ignore [valid-type]
        key,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return parse_obj[key]

    @overload  # type: ignore [no-redef]
    def prerprocess(
        self, parse_obj: is_table, key: None  # type: ignore [valid-type]  # pylint:disable=unused-argument
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return parse_obj

    @overload
    def apply(self, parse_obj):
        """Apply the parse function to a parser object."""
        raise NotImplementedError

    @overload  # type: ignore [no-redef]
    def apply(self, parse_obj: dict):
        return self.parser_fn(parse_obj)

    @overload  # type: ignore [no-redef]
    def apply(self, parse_obj: is_field):  # type: ignore [valid-type]
        if self.parser.element_wise:
            return parse_obj.map(self.parser_fn)
        return self.parser_fn(parse_obj)

    @overload  # type: ignore [no-redef]
    def apply(self, parse_obj: is_table):  # type: ignore [valid-type]
        if self.parser.element_wise:
            return parse_obj.apply(self.parser_fn, axis=1)
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
        key: Optional[str] = None,
    ):
        parse_obj = self.prerprocess(parse_obj, key)
        try:
            parser_output = self.apply(parse_obj)
        except DispatchError as exc:
            if exc.__cause__ is not None:
                raise exc.__cause__
            raise exc
        return self.postprocess(parse_obj, parser_output)
