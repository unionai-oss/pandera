"""Core pandas dataframe container specification."""

import warnings
from typing import Optional

import pandas as pd

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes
from pandera.config import get_config_context
from pandera.engines import pandas_engine
from pandera.errors import BackendNotFoundError
from pandera.import_utils import strategy_import_error


class DataFrameSchema(_DataFrameSchema[pd.DataFrame]):
    """A lightweight pandas DataFrame validator."""

    @_DataFrameSchema.dtype.setter  # type: ignore[attr-defined]
    def dtype(self, value: PandasDtypeInputTypes) -> None:
        """Set the dtype property."""
        self._dtype = pandas_engine.Engine.dtype(value) if value else None

    def validate(
        self,
        check_obj: pd.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Validate a DataFrame based on the schema specification.

        :param pd.DataFrame check_obj: the dataframe to be validated.
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
        :returns: validated ``DataFrame``

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        Calling ``schema.validate`` returns the dataframe.

        >>> import pandas as pd
        >>> import pandera.pandas as pa
        >>>
        >>> df = pd.DataFrame({
        ...     "probability": [0.1, 0.4, 0.52, 0.23, 0.8, 0.76],
        ...     "category": ["dog", "dog", "cat", "duck", "dog", "dog"]
        ... })
        >>>
        >>> schema_withchecks = pa.DataFrameSchema({
        ...     "probability": pa.Column(
        ...         float, pa.Check(lambda s: (s >= 0) & (s <= 1))),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": pa.Column(
        ...         str, [
        ...             pa.Check(lambda s: s.isin(["dog", "cat", "duck"])),
        ...             pa.Check(lambda s: (s == "dog").mean() > 0.5),
        ...         ]),
        ... })
        >>>
        >>> schema_withchecks.validate(df)[["probability", "category"]]
           probability category
        0         0.10      dog
        1         0.40      dog
        2         0.52      cat
        3         0.23     duck
        4         0.80      dog
        5         0.76      dog
        """
        if not get_config_context().validation_enabled:
            return check_obj

        # NOTE: Move this into its own schema-backend variant. This is where
        # the benefits of separating the schema spec from the backend
        # implementation comes in.

        if hasattr(check_obj, "dask"):
            # special case for dask dataframes

            from pandera.accessors import dask_accessor

            if inplace:
                check_obj = check_obj.pandera.add_schema(self)
            else:
                check_obj = check_obj.copy()

            check_obj = check_obj.map_partitions(  # type: ignore [operator]
                self._validate,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                meta=check_obj,
            )
            return check_obj.pandera.add_schema(self)

        return self._validate(
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    def _validate(
        self,
        check_obj: pd.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        return self.get_backend(check_obj).validate(
            check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        from pandera.backends.pandas.register import register_pandas_backends

        _cls = check_obj_cls
        try:
            register_pandas_backends(f"{_cls.__module__}.{_cls.__name__}")
        except BackendNotFoundError:
            for base_cls in _cls.__bases__:
                base_cls_name = f"{base_cls.__module__}.{base_cls.__name__}"
                try:
                    register_pandas_backends(base_cls_name)
                except BackendNotFoundError:
                    pass

    ###########################
    # Schema Strategy Methods #
    ###########################

    @strategy_import_error
    def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
        """Create a ``hypothesis`` strategy for generating a DataFrame.

        :param size: number of elements to generate
        :param n_regex_columns: number of regex columns to generate.
        :returns: a strategy that generates pandas DataFrame objects.
        """
        import pandera.strategies.pandas_strategies as st

        self.register_default_backends(pd.DataFrame)

        return st.dataframe_strategy(
            self.dtype,
            columns=self.columns,
            checks=self.checks,
            unique=self.unique,
            index=self.index,
            size=size,
            n_regex_columns=n_regex_columns,
        )

    def example(
        self, size: int | None = None, n_regex_columns: int = 1
    ) -> pd.DataFrame:
        """Generate an example of a particular size.

        :param size: number of elements in the generated DataFrame.
        :returns: pandas DataFrame object.
        """

        import hypothesis

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return self.strategy(
                size=size, n_regex_columns=n_regex_columns
            ).example()
