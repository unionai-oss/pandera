"""Parser backend for pandas"""

from functools import partial
from typing import Any, Optional, Union

import pandas as pd

from pandera.api.base.parsers import ParserResult
from pandera.api.pandas.types import is_field, is_table
from pandera.api.parsers import Parser
from pandera.backends.base import BaseParserBackend
from pandera.errors import ParserSourceError, ParserTargetError


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
            return parse_obj.map(self.parser_fn)
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

    def check_source(self, parse_obj: pd.DataFrame) -> None:
        """Verify the declared source columns are present before calling out.

        Without this a parser reading an absent column fails with a bare
        ``KeyError`` that names neither the parser nor the schema.
        """
        missing = [
            column
            for column in self.parser.source or ()
            if column not in parse_obj.columns
        ]
        if missing:
            available = list(parse_obj.columns)
            raise ParserSourceError(
                f"parser '{self.parser.name}' declares source column(s) "
                f"{missing} which are not in the dataframe. "
                f"Columns in dataframe: {available}",
                failure_cases=missing,
            )

    def apply_derivation(self, parse_obj: pd.DataFrame) -> pd.DataFrame:
        """Run a parser that declares the columns it reads and/or produces."""
        source = self.parser.source
        target = self.parser.target

        if source is None or getattr(self.parser, "frame_input", False):
            parser_input: Union[pd.Series, pd.DataFrame] = (
                parse_obj if source is None else parse_obj[list(source)]
            )
        elif len(source) == 1 and target is not None and len(target) == 1:
            # single column in, single column out: the ergonomic case, where
            # the function is written Series -> Series.
            parser_input = parse_obj[source[0]]
        else:
            parser_input = parse_obj[list(source)]

        if self.parser.element_wise:
            parser_output = parser_input.map(self.parser_fn)
        else:
            parser_output = self.parser_fn(parser_input)

        return self.assign_target(parse_obj, parser_output)

    def assign_target(
        self,
        parse_obj: pd.DataFrame,
        parser_output: Any,
    ) -> pd.DataFrame:
        """Place a parser's output into its declared target columns."""
        target = self.parser.target
        if target is None:
            if not isinstance(parser_output, pd.DataFrame):
                raise ParserTargetError(
                    f"parser '{self.parser.name}' declares a source but no "
                    "target, so it must return a DataFrame; got "
                    f"{type(parser_output).__name__}."
                )
            return parser_output

        parse_obj = parse_obj.copy()

        if len(target) == 1 and not isinstance(parser_output, pd.DataFrame):
            parse_obj[target[0]] = parser_output
            return parse_obj

        if not isinstance(parser_output, pd.DataFrame):
            raise ParserTargetError(
                f"parser '{self.parser.name}' declares targets {list(target)} "
                f"so it must return a DataFrame; got "
                f"{type(parser_output).__name__}."
            )

        missing = [col for col in target if col not in parser_output.columns]
        if missing:
            raise ParserTargetError(
                f"parser '{self.parser.name}' did not produce its declared "
                f"target column(s) {missing}. Columns returned: "
                f"{list(parser_output.columns)}",
                failure_cases=missing,
            )
        for column in target:
            parse_obj[column] = parser_output[column]
        # Only the target columns are taken from the parser's output, so any
        # frame-level metadata it attached would otherwise be dropped here.
        parse_obj.attrs.update(parser_output.attrs)
        return parse_obj

    def __call__(
        self,
        parse_obj: Union[pd.Series, pd.DataFrame],
        key: str | None = None,
    ):
        if self.parser.derives_columns and is_table(parse_obj):
            self.check_source(parse_obj)
            parser_output = self.apply_derivation(parse_obj)
            return self.postprocess(parse_obj, parser_output)

        parse_obj = self.preprocess(parse_obj, key)
        parser_output = self.apply(parse_obj)
        return self.postprocess(parse_obj, parser_output)
