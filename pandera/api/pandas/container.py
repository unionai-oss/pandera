"""Core pandas dataframe container specification."""

import warnings
from typing import Optional

import pandas as pd

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes
from pandera.backends.pandas.register import register_pandas_backends
from pandera.config import get_config_context
from pandera.dtypes import DataType
from pandera.engines import pandas_engine


# pylint: disable=too-many-public-methods,too-many-locals
class DataFrameSchema(_DataFrameSchema[pd.DataFrame]):
    """A light-weight pandas DataFrame validator."""

    def _register_default_backends(self):
        register_pandas_backends()

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: PandasDtypeInputTypes) -> None:
        """Set the pandas dtype property."""
        self._dtype = pandas_engine.Engine.dtype(value) if value else None

    def validate(
        self,
        check_obj: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
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
        >>> import pandera as pa
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
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        if self._is_inferred:
            warnings.warn(
                f"This {type(self)} is an inferred schema that hasn't been "
                "modified. It's recommended that you refine the schema "
                "by calling `add_columns`, `remove_columns`, or "
                "`update_columns` before using it to validate data.",
                UserWarning,
            )

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
