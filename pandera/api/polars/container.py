"""DataFrame Schema for Polars."""

from __future__ import annotations

import os
import sys
import warnings
from typing import TYPE_CHECKING, overload

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.polars.types import PolarsCheckObjects, PolarsFrame
from pandera.api.polars.utils import get_validation_depth
from pandera.backends.polars.register import register_polars_backends
from pandera.config import config_context, get_config_context
from pandera.engines import polars_engine

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl


class DataFrameSchema(_DataFrameSchema[PolarsCheckObjects]):
    """A Polars LazyFrame or DataFrame validator."""

    def _validate_attributes(self):
        super()._validate_attributes()

        if self.unique_column_names:
            warnings.warn(
                "unique_column_names=True will have no effect on validation "
                "since polars DataFrames do not support duplicate column "
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
        check_obj_cls: type,
    ):
        register_polars_backends()

    def validate(
        self,
        check_obj: PolarsFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
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

    @overload  # type: ignore[override]
    def coerce_dtype(self, check_obj: pl.LazyFrame) -> pl.LazyFrame: ...
    @overload
    def coerce_dtype(self, check_obj: pl.DataFrame) -> pl.DataFrame: ...
    def coerce_dtype(self, check_obj):
        return super().coerce_dtype(check_obj)

    @_DataFrameSchema.dtype.setter  # type: ignore[attr-defined]
    def dtype(self, value) -> None:
        """Set the dtype property."""
        self._dtype = polars_engine.Engine.dtype(value) if value else None

    def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
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

    def example(self, size: int | None = None, n_regex_columns: int = 1):
        """Generate an example of a particular size.

        :param size: number of elements in the generated DataFrame.
        :returns: pandas DataFrame object.

        .. warning::

           This method is not implemented in polars backend.
        """
        raise NotImplementedError(
            "Data synthesis is not supported in with polars schemas."
        )

    #####################
    # Schema IO Methods #
    #####################

    @classmethod
    def from_yaml(cls, yaml_schema) -> Self:
        """Load schema from YAML (see :mod:`pandera.io.polars_io`)."""
        from pandera.io import polars_io

        return polars_io.from_yaml(yaml_schema)

    def to_yaml(
        self, stream: os.PathLike | None = None, *, minimal: bool = True
    ) -> str | None:
        """Write schema to YAML (see :mod:`pandera.io.polars_io`)."""
        from pandera.io import polars_io

        return polars_io.to_yaml(self, stream=stream, minimal=minimal)

    @classmethod
    def from_json(cls, source) -> Self:
        """Load schema from JSON (see :mod:`pandera.io.polars_io`)."""
        from pandera.io import polars_io

        return polars_io.from_json(source)

    @overload
    def to_json(
        self, target: None = None, *, minimal: bool = True, **kwargs
    ) -> str:  # pragma: no cover
        ...

    @overload
    def to_json(
        self, target: os.PathLike, *, minimal: bool = True, **kwargs
    ) -> None:  # pragma: no cover
        ...

    def to_json(
        self,
        target: os.PathLike | None = None,
        *,
        minimal: bool = True,
        **kwargs,
    ) -> str | None:
        """Write schema to JSON (see :mod:`pandera.io.polars_io`)."""
        from pandera.io import polars_io

        return polars_io.to_json(self, target, minimal=minimal, **kwargs)
