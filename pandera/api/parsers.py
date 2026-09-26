"""Data validation parse definition."""

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

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
