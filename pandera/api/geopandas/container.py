"""GeoPandas :class:`geopandas.GeoDataFrame` schema API."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import cast, overload

import pandas as pd

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from pandera.api.base.types import StrictType
from pandera.api.geopandas.common import require_geopandas, to_geodataframe
from pandera.api.pandas.container import DataFrameSchema
from pandera.config import get_config_context
from pandera.import_utils import strategy_import_error


class GeoDataFrameSchema(DataFrameSchema):
    """Subclass of :class:`~pandera.api.pandas.container.DataFrameSchema` that
    returns a :class:`geopandas.GeoDataFrame` from :meth:`validate`,
    :meth:`example`, and :meth:`strategy`.

    Use the same constructor arguments as :class:`DataFrameSchema` (``columns``,
    ``checks``, ``index``, etc.). Validation uses the pandas backend; the result
    is coerced to a ``GeoDataFrame`` so geometry columns and CRS metadata are
    preserved for geospatial workflows.

    Requires the ``geopandas`` extra.
    """

    @classmethod
    def _from_dataframe_schema(cls, schema: DataFrameSchema) -> Self:
        """Construct a :class:`GeoDataFrameSchema` from a
        :class:`DataFrameSchema`."""
        return cls(
            columns=schema.columns,
            checks=schema.checks,
            parsers=schema.parsers,
            index=schema.index,
            dtype=schema.dtype,
            coerce=schema.coerce,
            strict=cast(StrictType, schema.strict),
            name=schema.name,
            ordered=schema.ordered,
            unique=schema.unique,
            report_duplicates=schema.report_duplicates,
            unique_column_names=schema.unique_column_names,
            add_missing_columns=schema.add_missing_columns,
            title=schema.title,
            description=schema.description,
            metadata=schema.metadata,
            drop_invalid_rows=schema.drop_invalid_rows,
        )

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
        """Like :meth:`DataFrameSchema.validate`, but returns a ``GeoDataFrame``
        when the result is a plain :class:`pandas.DataFrame`.

        If validation returns a Dask object (e.g. Dask DataFrame), it is passed
        through unchanged so distributed workflows are not broken.
        """
        if not get_config_context().validation_enabled:
            return check_obj

        result = super().validate(
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )
        if hasattr(result, "dask"):
            return result
        return to_geodataframe(result)

    @strategy_import_error
    def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
        """Like :meth:`DataFrameSchema.strategy`, but generated frames are
        ``GeoDataFrame`` instances when applicable.
        """
        require_geopandas()
        strat = super().strategy(size=size, n_regex_columns=n_regex_columns)
        return strat.map(to_geodataframe)

    def example(
        self, size: int | None = None, n_regex_columns: int = 1
    ) -> pd.DataFrame:
        """Like :meth:`DataFrameSchema.example`, but returns a ``GeoDataFrame``
        when applicable.
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

    #####################
    # Schema IO Methods #
    #####################

    def to_script(
        self, fp: str | Path | None = None, *, minimal: bool = True
    ) -> str | None:
        """Write :class:`GeoDataFrameSchema` to a Python script."""
        from pandera.io import pandas_io

        return pandas_io.to_script(self, fp, minimal=minimal)

    @classmethod
    def from_yaml(cls, yaml_schema) -> Self:
        """Load a :class:`GeoDataFrameSchema` from YAML."""
        schema = DataFrameSchema.from_yaml(yaml_schema)
        return cls._from_dataframe_schema(schema)

    def to_yaml(
        self,
        stream: os.PathLike | None = None,
        dataframe_library: str | None = None,
        *,
        minimal: bool = True,
    ) -> str | None:
        """Write schema to YAML (see :mod:`pandera.io.pandas_io`)."""
        from pandera.io import pandas_io

        return pandas_io.to_yaml(
            self,
            stream=stream,
            dataframe_library=dataframe_library,
            minimal=minimal,
        )

    @classmethod
    def from_json(cls, source) -> Self:
        """Load a :class:`GeoDataFrameSchema` from JSON."""
        schema = DataFrameSchema.from_json(source)
        return cls._from_dataframe_schema(schema)

    @overload
    def to_json(
        self,
        target: None = None,
        dataframe_library: str | None = None,
        *,
        minimal: bool = True,
        **kwargs,
    ) -> str: ...

    @overload
    def to_json(
        self,
        target: os.PathLike,
        dataframe_library: str | None = None,
        *,
        minimal: bool = True,
        **kwargs,
    ) -> None: ...

    def to_json(
        self,
        target: os.PathLike | None = None,
        dataframe_library: str | None = None,
        *,
        minimal: bool = True,
        **kwargs,
    ) -> str | None:
        """Write schema to JSON (see :mod:`pandera.io.pandas_io`)."""
        from pandera.io import pandas_io

        return pandas_io.to_json(
            self,
            target,
            dataframe_library=dataframe_library,
            minimal=minimal,
            **kwargs,
        )
