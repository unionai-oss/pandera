"""Data validation parse definition."""

import dataclasses
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

from pandera.api.base.parsers import BaseParser, ParserResult
from pandera.errors import SchemaInitError


def _as_column_tuple(
    columns: Union[str, Sequence[str], None],
    argument: str,
    parser_name: str | None,
) -> tuple[str, ...] | None:
    """Normalize a ``source``/``target`` argument to a tuple of column names."""
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = (columns,)
    columns = tuple(columns)
    if not columns:
        raise SchemaInitError(
            f"parser '{parser_name}' declares an empty {argument}. Pass "
            f"column name(s), or omit {argument} entirely."
        )
    non_strings = [col for col in columns if not isinstance(col, str)]
    if non_strings:
        raise SchemaInitError(
            f"parser '{parser_name}' {argument} must be column names, got "
            f"{non_strings!r}."
        )
    duplicates = sorted({col for col in columns if columns.count(col) > 1})
    if duplicates:
        raise SchemaInitError(
            f"parser '{parser_name}' {argument} has duplicate columns: "
            f"{duplicates}."
        )
    return columns


@dataclasses.dataclass(frozen=True)
class ParseContext:
    """Everything a :class:`ColumnParser` knows about the column it fills.

    Passed to ``bind`` at schema-build time, which is what lets a parser read
    its own column's declared type rather than having to be told what it is
    producing.
    """

    target: str
    """Name of the column being produced."""

    dtype: Any
    """The column's declared data type, or ``None`` if it has none."""

    description: str | None
    """The column's description."""

    nullable: bool
    """Whether the column accepts nulls."""

    checks: tuple[Any, ...]
    """The column's checks."""

    source: tuple[str, ...] | None
    """Columns this parser reads, resolved against the schema-wide default."""

    schema: Any
    """The schema being built, for cross-column resolution."""

    on_error: str = "raise"
    """What to do when the parser fails: ``raise``, ``null`` or ``drop``."""


@runtime_checkable
class ColumnParser(Protocol):
    """An object that knows how to produce a column.

    ``ParsedColumn(parser=...)`` accepts a plain callable or an object
    implementing this protocol. The protocol exists so a parser can inspect
    the column it is filling (via ``bind``) and so independent parsers that
    would otherwise each do their own work can be merged (via ``batch_key``).
    """

    def bind(self, ctx: ParseContext) -> Callable:
        """Return the function that produces this column.

        Called once at schema-build time. Raising
        :class:`~pandera.errors.SchemaInitError` here surfaces the problem
        before any data is touched.
        """
        ...  # pragma: no cover

    def batch_key(self, ctx: ParseContext) -> Hashable:
        """Group key for merging with sibling parsers, or ``None`` to opt out.

        Parsers returning equal non-``None`` keys are handed to ``batch`` and
        compiled into a single :class:`Parser` covering all their targets.
        """
        ...  # pragma: no cover


def _parser_name(column_parser: Any, targets: Sequence[str]) -> str:
    return f"{type(column_parser).__name__.lower()}[{','.join(targets)}]"


def _single_parser(column_parser: ColumnParser, ctx: ParseContext) -> "Parser":
    return Parser(
        column_parser.bind(ctx),
        source=list(ctx.source) if ctx.source else None,
        target=ctx.target,
        frame_input=getattr(column_parser, "frame_input", False),
        name=_parser_name(column_parser, [ctx.target]),
    )


def _batched_parser(
    items: Sequence[tuple[ColumnParser, ParseContext]],
) -> "Parser":
    """Compile a group of batchable column parsers into one ``Parser``."""
    first = items[0][0]
    batch = getattr(type(first), "batch", None)
    if batch is None:
        raise SchemaInitError(
            f"{type(first).__name__} returns a batch_key but does not "
            "implement a `batch` classmethod, so its columns cannot be "
            "merged. Return None from batch_key to opt out of batching."
        )
    sources: list[str] = []
    for _, ctx in items:
        for column in ctx.source or ():
            if column not in sources:
                sources.append(column)
    targets = [ctx.target for _, ctx in items]
    return Parser(
        batch(items),
        source=sources or None,
        target=targets,
        frame_input=True,
        name=_parser_name(first, targets),
    )


def compile_column_parsers(schema: Any) -> list["Parser"]:
    """Collect a schema's parsers, including those declared on its columns.

    Columns that declare their own derivation contribute a :class:`Parser`
    each, except where siblings share a ``batch_key`` and are merged into one.
    The result is ordered by dependency.
    """
    parsers: list[Parser] = list(schema.parsers)
    batches: dict[Hashable, list[tuple[ColumnParser, ParseContext]]] = {}

    for name, column in getattr(schema, "columns", {}).items():
        build = getattr(column, "build_parse_context", None)
        if build is None:
            continue
        ctx = build(name, schema)
        column_parser = column.parser

        if not hasattr(column_parser, "bind"):
            # a plain callable
            parsers.append(column.compile_parser(ctx))
            continue

        key = column_parser.batch_key(ctx)
        if key is None:
            parsers.append(_single_parser(column_parser, ctx))
        else:
            batches.setdefault(key, []).append((column_parser, ctx))

    # A non-None batch_key always goes through ``batch``, even for a group of
    # one, so a parser that opts into batching has exactly one code path.
    for items in batches.values():
        parsers.append(_batched_parser(items))

    return order_parsers(parsers)


def order_parsers(parsers: Sequence["Parser"]) -> list["Parser"]:
    """Order parsers so each runs after the parsers producing its sources.

    Parsers that declare neither a source nor a target keep their list
    position and run first, preserving the behavior of schemas written before
    derivation existed. Declared parsers are then emitted in dependency order,
    falling back to list order between independent ones so the result is
    deterministic.

    :raises SchemaInitError: if the declared dependencies contain a cycle.
    """
    undeclared = [p for p in parsers if not p.derives_columns]
    declared = [p for p in parsers if p.derives_columns]
    if len(declared) < 2:
        return [*undeclared, *declared]

    # column -> indices of the parsers that produce it
    producers: dict[str, list[int]] = {}
    for index, parser in enumerate(declared):
        for column in parser.target or ():
            producers.setdefault(column, []).append(index)

    dependencies: dict[int, set[int]] = {
        index: set() for index in range(len(declared))
    }
    for index, parser in enumerate(declared):
        for column in parser.source or ():
            for producer in producers.get(column, []):
                if producer != index:
                    dependencies[index].add(producer)

    ordered: list[int] = []
    emitted: set[int] = set()
    remaining = list(range(len(declared)))
    while remaining:
        ready = [i for i in remaining if dependencies[i] <= emitted]
        if not ready:
            cycle = sorted(
                column
                for i in remaining
                for column in (declared[i].target or ())
            )
            raise SchemaInitError(
                "parser dependencies form a cycle involving column(s) "
                f"{cycle}. A derived column cannot depend, directly or "
                "indirectly, on itself."
            )
        # ``ready`` preserves list order, so independent parsers stay stable.
        for index in ready:
            ordered.append(index)
            emitted.add(index)
            remaining.remove(index)

    return [*undeclared, *(declared[i] for i in ordered)]


class Parser(BaseParser):
    """Parse a data object for certain properties."""

    def __init__(
        self,
        parser_fn: Callable,
        element_wise: bool = False,
        ignore_na: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        source: Union[str, Sequence[str], None] = None,
        target: Union[str, Sequence[str], None] = None,
        frame_input: bool = False,
        **parser_kwargs,
    ) -> None:
        """Apply a parser function to a data object.

        :param parse_fn: A function to parser pandas data structure. For Column
            or SeriesSchema parsers, if element_wise is True, this function
            should have the signature: ``Callable[[pd.Series],
            Union[pd.Series, bool]]``, where the output series is a boolean
            vector.

            If element_wise is False, this function should have the signature:
            ``Callable[[Any], bool]``, where ``Any`` is an element in the
            column.

            For DataFrameSchema parsers, if element_wise=True, fn
            should have the signature: ``Callable[[pd.DataFrame],
            Union[pd.DataFrame, pd.Series, bool]]``, where the output dataframe
            or series contains booleans.

            If element_wise is True, fn is applied to each row in
            the dataframe with the signature ``Callable[[pd.Series], bool]``
            where the series input is a row in the dataframe.
        :param element_wise: Whether or not to apply validator in an
            element-wise fashion. If bool, assumes that all parsers should be
            applied to the column element-wise. If list, should be the same
            number of elements as parsers.
        :param name: optional name for the parser.
        :param title: A human-readable label for the parser.
        :param description: An arbitrary textual description of the parser.
        :param source: column(s) this parser reads. Declaring them lets pandera
            check they are present *before* calling ``parser_fn``, raising
            :class:`~pandera.errors.ParserSourceError` naming the parser and
            the missing column instead of a bare ``KeyError``.
        :param target: column(s) this parser produces. Declaring them lets
            pandera order parsers by their dependencies rather than by list
            position, and verify the function produced what it promised
            (:class:`~pandera.errors.ParserTargetError`).
        :param frame_input: always call ``parser_fn`` with a ``DataFrame``,
            even when a single source and target are declared. Used by
            batched column parsers, which fill several columns from one call
            and so cannot take the ``Series -> Series`` shortcut.
        :param parse_kwargs: key-word arguments to pass into ``parse_fn``

        When ``source`` and ``target`` are both a single column, ``parser_fn``
        is called with that source column as a ``Series`` and is expected to
        return a ``Series``. Otherwise it is called with the dataframe (or the
        selected source columns) and is expected to return a dataframe
        containing the targets.

        See :ref:`here<parsers>` for more usage details.

        """
        super().__init__(name=name)
        self._parser_fn = parser_fn
        self._parser_kwargs = parser_kwargs
        self.element_wise = element_wise
        self.ignore_na = ignore_na
        self.name = name or getattr(
            self._parser_fn, "__name__", self._parser_fn.__class__.__name__
        )
        self.title = title
        self.description = description
        self.source = _as_column_tuple(source, "source", self.name)
        self.target = _as_column_tuple(target, "target", self.name)
        self.frame_input = frame_input

    @property
    def derives_columns(self) -> bool:
        """Whether this parser declares what it reads or produces."""
        return self.source is not None or self.target is not None

    def __call__(
        self, parse_obj: Any, column: str | None = None
    ) -> ParserResult:
        """Validate pandas DataFrame or Series.

        :param parse_obj: pandas DataFrame of Series to validate.
        :param column: for dataframe parsers, apply the parser function to this
            column.
        :returns: ParseResult tuple containing:

            ``parser_output``: boolean scalar, ``Series`` or ``DataFrame``
            indicating which elements passed the parser.

            ``parsed_object``: the parseed object itself. Depending on the
            options provided to the ``Parse``, this will be a pandas Series,
            DataFrame, or if the ``groupby`` option is specified, a
            ``Dict[str, Series]`` or ``Dict[str, DataFrame]`` where the keys
            are distinct groups.

        """
        backend = self.get_backend(parse_obj)(self)
        return backend(parse_obj, column)
