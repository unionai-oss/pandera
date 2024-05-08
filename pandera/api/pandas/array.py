"""Core pandas array specification."""

import warnings
from typing import Any, Optional, cast

import pandas as pd

from pandera import errors
from pandera import strategies as st
from pandera.api.base.types import CheckList, ParserList
from pandera.api.dataframe.components import ComponentSchema, TDataObject
from pandera.api.pandas.types import PandasDtypeInputTypes, is_field
from pandera.backends.pandas.register import register_pandas_backends
from pandera.config import get_config_context
from pandera.dtypes import DataType, UniqueSettings
from pandera.engines import pandas_engine


class ArraySchema(ComponentSchema[TDataObject]):
    """Base array validator object."""

    def _validate_attributes(self):
        for check in self.checks:
            if check.groupby is not None and not self._allow_groupby:
                raise errors.SchemaInitError(
                    f"Cannot use groupby checks with type {type(self)}"
                )

        if isinstance(self.dtype, pandas_engine.PydanticModel):
            raise errors.SchemaInitError(
                "PydanticModel dtype can only be specified as a "
                "DataFrameSchema dtype."
            )

    def _register_default_backends(self):
        register_pandas_backends()

    @property
    def dtype(self) -> DataType:
        """Get the pandas dtype"""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: Optional[PandasDtypeInputTypes]) -> None:
        """Set the pandas dtype"""
        self._dtype = pandas_engine.Engine.dtype(value) if value else None

    ###########################
    # Schema Strategy Methods #
    ###########################

    @st.strategy_import_error
    def strategy(self, *, size=None):
        """Create a ``hypothesis`` strategy for generating a Series.

        :param size: number of elements to generate
        :returns: a strategy that generates pandas Series objects.
        """
        return st.series_strategy(
            self.dtype,
            checks=self.checks,
            nullable=self.nullable,
            unique=self.unique,
            name=self.name,
            size=size,
        )

    def example(self, size=None) -> TDataObject:
        """Generate an example of a particular size.

        :param size: number of elements in the generated array.
        :returns: array object.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(size=size).example()


class SeriesSchema(ArraySchema[pd.Series]):
    """A pandas Series validator."""

    def __init__(
        self,
        dtype: Optional[PandasDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        parsers: Optional[ParserList] = None,
        index=None,
        nullable: bool = False,
        unique: bool = False,
        report_duplicates: UniqueSettings = "all",
        coerce: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        metadata: Optional[dict] = None,
        drop_invalid_rows: bool = False,
    ) -> None:
        """Initialize series schema base object.

        :param dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:

            ``Callable[Any, bool]`` where the ``Any`` input is a scalar element
            in the column. Otherwise, the input is assumed to be a
            pandas.Series object.
        :param index: specify the datatypes and properties of the index.
        :param nullable: Whether or not column can contain null values.
        :param unique: Whether or not column can contain duplicate
            values.
        :param report_duplicates: how to report unique errors
            - `exclude_first`: report all duplicates except first occurence
            - `exclude_last`: report all duplicates except last occurence
            - `all`: (default) report all duplicates
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param name: series name.
        :param title: A human-readable label for the series.
        :param description: An arbitrary textual description of the series.
        :param metadata: An optional key-value data.
        :param default: The default value for missing values in the series.
        :param drop_invalid_rows: if True, drop invalid rows on validation.

        """
        super().__init__(
            dtype,
            checks,
            parsers,
            nullable,
            unique,
            report_duplicates,
            coerce,
            name,
            title,
            description,
            default,
            metadata,
            drop_invalid_rows,
        )
        self.index = index

    @property
    def _allow_groupby(self) -> bool:
        """Whether the schema or schema component allows groupby operations."""
        return False

    def validate(  # type: ignore [override]
        self,
        check_obj: pd.Series,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.Series:
        """Validate a Series object.

        :param check_obj: One-dimensional ndarray with axis labels
            (including time series).
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated Series.

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>> series_schema = pa.SeriesSchema(
        ...     float, [
        ...         pa.Check(lambda s: s > 0),
        ...         pa.Check(lambda s: s < 1000),
        ...         pa.Check(lambda s: s.mean() > 300),
        ...     ])
        >>> series = pd.Series([1, 100, 800, 900, 999], dtype=float)
        >>> print(series_schema.validate(series))
        0      1.0
        1    100.0
        2    800.0
        3    900.0
        4    999.0
        dtype: float64

        """
        if not get_config_context().validation_enabled:
            return check_obj

        if self._is_inferred:
            warnings.warn(
                f"This {type(self)} is an inferred schema that hasn't been "
                "modified. It's recommended that you refine the schema "
                "by calling `set_checks` before using it to validate data.",
                UserWarning,
            )

        if not is_field(check_obj):
            raise TypeError(f"expected pd.Series, got {type(check_obj)}")

        if hasattr(check_obj, "dask"):
            # special case for dask series
            if inplace:
                check_obj = check_obj.pandera.add_schema(self)
            else:
                check_obj = check_obj.copy()

            check_obj = check_obj.map_partitions(
                super().validate,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                meta=check_obj,
            )
            check_obj = check_obj.pandera.add_schema(self)
            return cast(pd.Series, check_obj)

        validated_obj = super().validate(
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )
        if self.index is not None:
            validated_obj = self.index.validate(
                check_obj,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
        return cast(pd.Series, validated_obj)

    def example(self, size=None) -> pd.Series:
        """Generate an example of a particular size.

        :param size: number of elements in the generated Series.
        :returns: pandas Series object.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,import-error
        return cast(pd.Series, super().example(size=size))
