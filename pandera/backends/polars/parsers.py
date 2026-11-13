"""Parser backend for polars."""

from functools import partial
from typing import Any, Union

import polars as pl

from pandera.api.base.parsers import ParserResult
from pandera.api.parsers import Parser
from pandera.backends.base import BaseParserBackend
from pandera.errors import ParserSourceError, ParserTargetError


class PolarsParserBackend(BaseParserBackend):
    """Parser backend of polars."""

    def __init__(self, parser: Parser):
        """Initializes a parser backend object."""
        super().__init__(parser)
        assert parser._parser_fn is not None, "Parser._parser_fn must be set."
        self.parser = parser
        self.parser_fn = partial(parser._parser_fn, **parser._parser_kwargs)

    def check_source(self, parse_obj: pl.DataFrame) -> None:
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

    def apply_derivation(self, parse_obj: pl.DataFrame) -> pl.DataFrame:
        """Run a parser that declares the columns it reads and/or produces."""
        source = self.parser.source
        target = self.parser.target

        if source is None or getattr(self.parser, "frame_input", False):
            parser_input: Union[pl.Series, pl.DataFrame] = (
                parse_obj if source is None else parse_obj.select(list(source))
            )
        elif len(source) == 1 and target is not None and len(target) == 1:
            # single column in, single column out: the ergonomic case, where
            # the function is written Series -> Series.
            parser_input = parse_obj[source[0]]
        else:
            parser_input = parse_obj.select(list(source))

        if self.parser.element_wise:
            parser_output = parser_input.map_elements(  # type: ignore[union-attr]
                self.parser_fn
            )
        else:
            parser_output = self.parser_fn(parser_input)

        return self.assign_target(parse_obj, parser_output)

    def assign_target(
        self,
        parse_obj: pl.DataFrame,
        parser_output: Any,
    ) -> pl.DataFrame:
        """Place a parser's output into its declared target columns."""
        target = self.parser.target
        if target is None:
            if not isinstance(parser_output, pl.DataFrame):
                raise ParserTargetError(
                    f"parser '{self.parser.name}' declares a source but no "
                    "target, so it must return a DataFrame; got "
                    f"{type(parser_output).__name__}."
                )
            return parser_output

        if len(target) == 1 and not isinstance(parser_output, pl.DataFrame):
            if isinstance(parser_output, pl.Series):
                return parse_obj.with_columns(parser_output.alias(target[0]))
            return parse_obj.with_columns(pl.Series(target[0], parser_output))

        if not isinstance(parser_output, pl.DataFrame):
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
        return parse_obj.with_columns(
            [parser_output[column] for column in target]
        )

    def preprocess(
        self, parse_obj: Any, key: str | None
    ) -> pl.DataFrame | pl.Series:
        """Preprocesses a parser object before applying the parse function."""
        if isinstance(parse_obj, pl.DataFrame):
            if key is not None:
                # a single-column parser operates on the column Series, mirroring
                # the pandas field-parser behavior.
                return parse_obj[key]
            return parse_obj
        return parse_obj

    def apply(self, parse_obj: Any) -> Any:
        """Apply the parse function to a parser object."""
        if self.parser.element_wise:
            return parse_obj.map_elements(self.parser_fn)  # type: ignore[union-attr]
        return self.parser_fn(parse_obj)

    def postprocess(
        self,
        parse_obj: Any,
        parser_output: Any,
    ) -> ParserResult:
        """Postprocesses the result of applying the parser function."""
        return ParserResult(
            parser_output=parser_output, parsed_object=parse_obj
        )

    def __call__(
        self,
        parse_obj: Union[pl.DataFrame, pl.Series],
        key: str | None = None,
    ):
        if self.parser.derives_columns and isinstance(parse_obj, pl.DataFrame):
            self.check_source(parse_obj)
            parser_output = self.apply_derivation(parse_obj)
            return self.postprocess(parse_obj, parser_output)

        parse_obj = self.preprocess(parse_obj, key)
        parser_output = self.apply(parse_obj)
        return self.postprocess(parse_obj, parser_output)
