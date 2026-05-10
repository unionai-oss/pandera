"""Register pandas backends."""

from functools import lru_cache
from typing import Optional

from pandera.backends.pandas.array import SeriesSchemaBackend
from pandera.backends.pandas.checks import PandasCheckBackend
from pandera.backends.pandas.components import (
    ColumnBackend,
    IndexBackend,
    MultiIndexBackend,
)
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.backends.pandas.hypotheses import PandasHypothesisBackend
from pandera.backends.pandas.parsers import PandasParserBackend


@lru_cache
def register_pandas_backends(
    check_cls_fqn: str | None = None,
):
    """Register pandas backends.

    This function is called at schema initialization in the _register_*_backends
    method.

    When ``PANDERA_USE_NARWHALS_BACKEND=True`` (or
    ``pandera.config.CONFIG.use_narwhals_backend`` is ``True``) this function
    additionally calls :func:`register_pandas_via_narwhals` so that
    ``pd.DataFrame`` can be validated against ``pandera.polars`` /
    ``pandera.ibis`` ``DataFrameSchema``s through the Narwhals backend. The
    native pandas backends are always registered;
    ``pandera.pandas.DataFrameSchema`` continues to dispatch through them.

    :param framework_name: name of the framework to register backends for.
        Allowable types are "pandas", "dask", "modin", "pyspark", and
        "geopandas".
    """

    from pandera._patch_numpy2 import _patch_numpy2

    _patch_numpy2()

    # NOTE: This registers the deprecated DataFrameSchema class. Remove this
    # once the deprecated class is removed.
    from pandera._pandas_deprecated import (
        DataFrameSchema as _DataFrameSchemaDeprecated,
    )
    from pandera.api.checks import Check
    from pandera.api.geopandas.container import GeoDataFrameSchema
    from pandera.api.hypotheses import Hypothesis
    from pandera.api.pandas.array import SeriesSchema
    from pandera.api.pandas.components import Column, Index, MultiIndex
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.api.pandas.types import get_backend_types
    from pandera.api.parsers import Parser

    assert check_cls_fqn is not None, (
        "pandas backend registration requires passing in the fully qualified "
        "check class name"
    )
    backend_types = get_backend_types(check_cls_fqn)

    from pandera.backends.pandas import builtin_checks, builtin_hypotheses

    for t in backend_types.check_backend_types:
        Check.register_backend(t, PandasCheckBackend)
        Hypothesis.register_backend(t, PandasHypothesisBackend)
        Parser.register_backend(t, PandasParserBackend)

    for t in backend_types.dataframe_datatypes:
        DataFrameSchema.register_backend(t, DataFrameSchemaBackend)
        # Same pandas backend; GeoDataFrameSchema only changes validate output.
        GeoDataFrameSchema.register_backend(t, DataFrameSchemaBackend)
        _DataFrameSchemaDeprecated.register_backend(t, DataFrameSchemaBackend)
        Column.register_backend(t, ColumnBackend)
        MultiIndex.register_backend(t, MultiIndexBackend)
        Index.register_backend(t, IndexBackend)

    for t in backend_types.series_datatypes:
        SeriesSchema.register_backend(t, SeriesSchemaBackend)
        Column.register_backend(t, ColumnBackend)
        MultiIndex.register_backend(t, MultiIndexBackend)
        Index.register_backend(t, IndexBackend)

    for t in backend_types.index_datatypes:
        Index.register_backend(t, IndexBackend)

    for t in backend_types.multiindex_datatypes:
        MultiIndex.register_backend(t, MultiIndexBackend)

    # When the Narwhals backend is opted in, additionally wire ``pd.DataFrame``
    # cross-backend support so ``pandera.polars`` / ``pandera.ibis`` schemas
    # accept pandas inputs. ``register_pandas_via_narwhals`` is idempotent and
    # a no-op when the flag is False.
    register_pandas_via_narwhals()


@lru_cache
def register_pandas_via_narwhals() -> None:
    """Register ``pd.DataFrame`` against polars/ibis schemas via Narwhals.

    Mirrors the ``CONFIG.use_narwhals_backend`` opt-in branch in
    :func:`pandera.backends.polars.register.register_polars_backends` and
    :func:`pandera.backends.ibis.register.register_ibis_backends`: when the
    flag is enabled, ``pd.DataFrame`` becomes a valid input to
    ``pandera.polars.DataFrameSchema`` and ``pandera.ibis.DataFrameSchema``
    (in addition to the native pandas backend, which is always registered
    by :func:`register_pandas_backends`). No-op when the flag is False.

    Decorated with ``@lru_cache`` to keep registration idempotent across
    repeated ``validate()`` calls. Tests that flip ``use_narwhals_backend``
    mid-session should call :meth:`cache_clear` on this function.

    Raises
    ------
    ImportError
        If the Narwhals backend is enabled but the ``narwhals`` package is
        not installed.
    """
    from pandera.config import CONFIG

    if not CONFIG.use_narwhals_backend:
        return

    try:
        import narwhals.stable.v1  # noqa: F401  (validate availability)
    except ImportError as exc:
        raise ImportError(
            "The Narwhals backend is enabled but the 'narwhals' "
            "package is not installed. Install it with: "
            "pip install 'pandera[narwhals]'"
        ) from exc

    import pandas as pd

    import pandera.backends.narwhals.builtin_checks  # noqa: F401
    from pandera.backends.narwhals.components import (
        ColumnBackend as NwColumnBackend,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NwDataFrameSchemaBackend,
    )

    # IMPORTANT: We deliberately do NOT register ``pd.DataFrame`` against
    # ``Check`` here. ``Check.register_backend`` is first-write-wins, and
    # the legacy pandas backend already claims ``pd.DataFrame``; letting
    # both compete would silently break pandera's own pandas tests. The
    # narwhals backend wraps every native input as an ``nw.LazyFrame``
    # before invoking checks, so dispatch via the registered
    # ``nw.LazyFrame`` / ``nw.DataFrame`` entries (set up in
    # ``register_polars_backends`` / ``register_ibis_backends``) is
    # sufficient for dataframe-level checks.
    try:
        from pandera.api.polars.components import Column as PolarsColumn
        from pandera.api.polars.container import (
            DataFrameSchema as PolarsDataFrameSchema,
        )
    except ImportError:
        pass
    else:
        PolarsDataFrameSchema.register_backend(
            pd.DataFrame, NwDataFrameSchemaBackend
        )
        PolarsColumn.register_backend(pd.DataFrame, NwColumnBackend)

    try:
        from pandera.api.ibis.components import Column as IbisColumn
        from pandera.api.ibis.container import (
            DataFrameSchema as IbisDataFrameSchema,
        )
    except ImportError:
        pass
    else:
        IbisDataFrameSchema.register_backend(
            pd.DataFrame, NwDataFrameSchemaBackend
        )
        IbisColumn.register_backend(pd.DataFrame, NwColumnBackend)
