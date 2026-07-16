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

# Fully qualified check-class names that have been registered in this process.
# Used by re_register_pandas_backends() to replay registrations after the
# use_narwhals_backend config flag is toggled at runtime.
_registered_check_cls_fqns: set[str] = set()


@lru_cache
def register_pandas_backends(
    check_cls_fqn: str | None = None,
    use_narwhals_backend: bool = False,
):
    """Register pandas backends.

    This function is called at schema initialization in the _register_*_backends
    method.

    The native pandas backends are always registered — they serve
    ``SeriesSchema``, ``Index``, ``MultiIndex``, ``Parser``, and ``Hypothesis``
    validation as well as the pandas-like frame types (dask, modin, geopandas,
    pyspark.pandas). When ``use_narwhals_backend`` is ``True`` (from
    ``PANDERA_USE_NARWHALS_BACKEND`` or ``pandera.config.CONFIG``), the
    ``pandas.DataFrame`` registry entries for ``DataFrameSchema`` and
    ``Column`` (plus Narwhals-frame ``Check`` dispatch) are overridden with
    the Narwhals backend implementations, mirroring
    :func:`pandera.backends.polars.register.register_polars_backends`.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls. The backend choice is part of the cache key; programmatic
    changes to ``CONFIG.use_narwhals_backend`` after registration trigger
    automatic re-registration via ``pandera.config.set_config``.

    :param check_cls_fqn: fully qualified name of the class of the object to
        be validated, e.g. "pandas.core.frame.DataFrame". Determines which
        framework's types (pandas, dask, modin, pyspark.pandas, geopandas)
        are registered.
    :param use_narwhals_backend: if True, route ``pd.DataFrame`` validation
        through the Narwhals backend.
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

    _registered_check_cls_fqns.add(check_cls_fqn)

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

    if use_narwhals_backend:
        _register_narwhals_pandas_overrides()


def _register_narwhals_pandas_overrides():
    """Route ``pd.DataFrame`` validation through the Narwhals backend.

    Overrides only the ``pandas.DataFrame`` registry entries for
    ``DataFrameSchema`` and ``Column`` (plus Narwhals-frame ``Check``
    dispatch). Everything else — ``SeriesSchema``, ``Index``, ``MultiIndex``,
    ``Parser``, ``Hypothesis``, and the other pandas-like frame types (dask,
    modin, geopandas, pyspark.pandas) — stays on the native pandas backends.
    """
    try:
        import narwhals.stable.v1 as nw
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The Narwhals backend is enabled but the 'narwhals' "
            "package is not installed. Install it with: "
            "pip install 'pandera[narwhals]'"
        ) from exc

    import pandas as pd

    import pandera.backends.narwhals.builtin_checks  # noqa: F401
    from pandera._pandas_deprecated import (
        DataFrameSchema as _DataFrameSchemaDeprecated,
    )
    from pandera.api.checks import Check
    from pandera.api.pandas.components import Column
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import (
        ColumnBackend as NarwhalsColumnBackend,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )

    DataFrameSchema.register_backend(
        pd.DataFrame, NarwhalsDataFrameSchemaBackend, force=True
    )
    _DataFrameSchemaDeprecated.register_backend(
        pd.DataFrame, NarwhalsDataFrameSchemaBackend, force=True
    )
    Column.register_backend(pd.DataFrame, NarwhalsColumnBackend, force=True)
    # pandas frames are eager under narwhals: the narwhals backend wraps them
    # as nw.LazyFrame for validation, so checks dispatch on both wrapper
    # types. The (Check, pd.DataFrame) entry intentionally stays on the
    # native PandasCheckBackend — direct Check calls on native pandas frames
    # are unchanged.
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend, force=True)
    Check.register_backend(nw.DataFrame, NarwhalsCheckBackend, force=True)


def re_register_pandas_backends(*, use_narwhals_backend: bool):
    """Re-register pandas backends after toggling ``use_narwhals_backend``.

    Unlike the polars/ibis/pyspark register functions,
    :func:`register_pandas_backends` is parameterized by the check class fqn,
    so re-registration replays every fqn registered so far in this process
    with the new flag value.
    """
    fqns = sorted(_registered_check_cls_fqns)
    register_pandas_backends.cache_clear()
    for fqn in fqns:
        register_pandas_backends(
            fqn, use_narwhals_backend=use_narwhals_backend
        )
