"""Data validation parse definition."""

from typing import Any, Callable, Optional

from pandera.api.base.parsers import BaseParser, ParserResult


# pylint: disable=too-many-public-methods
class Parser(BaseParser):
    """Parse a data object for certain properties."""

    def __init__(
        self,
        parser_fn: Callable,
        element_wise: bool = False,
        ignore_na: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
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
        :param parse_kwargs: key-word arguments to pass into ``parse_fn``

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

    def __call__(
        self, parse_obj: Any, column: Optional[str] = None
    ) -> ParserResult:
        # pylint: disable=too-many-branches
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
