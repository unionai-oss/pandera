"""DataFrame Schema for Polars."""

import warnings
from typing import Optional, Type

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.polars.types import PolarsCheckObjects, PolarsFrame
from pandera.api.polars.utils import get_validation_depth
from pandera.backends.polars.register import register_polars_backends
from pandera.config import config_context, get_config_context
from pandera.dtypes import DataType
from pandera.engines import polars_engine


class DataFrameSchema(_DataFrameSchema[PolarsCheckObjects]):
    """A polars LazyFrame or DataFrame validator."""

    def _validate_attributes(self):
        super()._validate_attributes()

        if self.unique_column_names:
            warnings.warn(
                "unique_column_names=True will have no effect on validation "
                "since polars DataFrames does not support duplicate column "
                "names."
            )

        if self.report_duplicates != "all":
            warnings.warn(
                "Setting report_duplicates to 'exclude_first' or "
                "'exclude_last' will have no effect on validation. With the "
                "polars backend, all duplicate values will be reported."
            )

    @staticmethod
    def register_default_backends(
        check_obj_cls: Type,
    ):  # pylint: disable=unused-argument
        register_polars_backends()

    def validate(
        self,
        check_obj: PolarsFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> PolarsFrame:
        """Validate a polars DataFrame against the schema."""

        if not get_config_context().validation_enabled:
            return check_obj

        with config_context(validation_depth=get_validation_depth(check_obj)):
            # if validating a polars DataFrame, use the global config setting
            output = self.get_backend(check_obj).validate(
                check_obj=check_obj,
                schema=self,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )

        return output

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value) -> None:
        """Set the pandas dtype property."""
        self._dtype = polars_engine.Engine.dtype(value) if value else None

    def strategy(
        self, *, size: Optional[int] = None, n_regex_columns: int = 1
    ):
        """Create a ``hypothesis`` strategy for generating a DataFrame.

        :param size: number of elements to generate
        :param n_regex_columns: number of regex columns to generate.
        :returns: a strategy that generates pandas DataFrame objects.

        .. warning::

           This method is not implemented in the polars backend.
        """
        raise NotImplementedError(
            "Data synthesis is not supported in with polars schemas."
        )

    def example(self, size: Optional[int] = None, n_regex_columns: int = 1):
        """Generate an example of a particular size.

        :param size: number of elements in the generated DataFrame.
        :returns: pandas DataFrame object.

        .. warning::

           This method is not implemented in polars backend.
        """
        raise NotImplementedError(
            "Data synthesis is not supported in with polars schemas."
        )
