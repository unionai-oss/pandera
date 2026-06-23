"""Lazy registration of check backends."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_get_backend_types_from_mro() -> Callable | None:
    """Imports ``get_backend_types_from_mro`` once, tolerating missing pandas.

    ``pandera.api.pandas.types`` imports numpy and pandas at module load, which
    raises in environments that only install one of the disjoint extras (e.g.
    ``pandera[polars]`` without numpy/pandas). The result is cached so the
    import (and any failure) is attempted once rather than on every check, since
    ``register_default_check_backends`` runs on the validation hot path and
    Python does not cache failed imports in ``sys.modules``.

    A non-pandas-absence ``ImportError`` here (e.g. a broken transitive
    dependency or a circular-import regression) is also swallowed. If a real
    pandas frame later surfaces a confusing "Backend not found" error, suspect
    this import. See GH #2387.

    Returns:
        The ``get_backend_types_from_mro`` callable, or ``None`` when pandas is
        not importable.
    """
    try:
        from pandera.api.pandas.types import get_backend_types_from_mro
    except ImportError:
        return None
    return get_backend_types_from_mro


def register_default_check_backends(check_obj_cls: type) -> None:
    """Register check backends for ``check_obj_cls`` at validation time."""
    module = getattr(check_obj_cls, "__module__", "")

    # Narwhals wrapper types share pandas-like class names (e.g. DataFrame)
    # but must not route through the pandas backend registry.
    if module.startswith("narwhals."):
        from pandera.config import CONFIG

        use_nw = CONFIG.use_narwhals_backend
        try:
            from pandera.backends.polars.register import (
                register_polars_backends,
            )

            register_polars_backends(use_narwhals_backend=use_nw)
        except ImportError:
            pass
        try:
            from pandera.backends.ibis.register import register_ibis_backends

            register_ibis_backends(use_narwhals_backend=use_nw)
        except ImportError:
            pass
        try:
            from pandera.backends.pyspark.register import (
                register_pyspark_backends,
            )

            register_pyspark_backends(use_narwhals_backend=use_nw)
        except ImportError:
            pass
        return

    # Consult the pandas-like backend types first so pandas-like backends
    # (notably ``pyspark.pandas``, whose module name starts with ``pyspark`` but
    # uses the pandas backends) keep routing correctly. The import is guarded so
    # pandas-free installs fall through to the native dispatch below. See #2387.
    get_backend_types_from_mro = _load_get_backend_types_from_mro()

    if (
        get_backend_types_from_mro is not None
        and get_backend_types_from_mro(check_obj_cls) is not None
    ):
        from pandera.api.pandas.container import (
            DataFrameSchema as PandasDataFrameSchema,
        )

        PandasDataFrameSchema.register_default_backends(check_obj_cls)
        return

    if module.startswith("polars."):
        from pandera.backends.polars.register import register_polars_backends
        from pandera.config import CONFIG

        register_polars_backends(
            use_narwhals_backend=CONFIG.use_narwhals_backend
        )
        return

    if module.startswith("ibis."):
        from pandera.backends.ibis.register import register_ibis_backends
        from pandera.config import CONFIG

        register_ibis_backends(
            use_narwhals_backend=CONFIG.use_narwhals_backend
        )
        return

    # Match ``pyspark.sql`` rather than bare ``pyspark`` so that
    # ``pyspark.pandas`` frames (module prefix ``pyspark.pandas``) are not
    # captured here. They route through the pandas-MRO check above when pandas
    # is importable; narrowing keeps them from misrouting to the pyspark-sql
    # backend in a pandas-free install where that check returns None. See #2387.
    if module.startswith("pyspark.sql"):
        from pandera.backends.pyspark.register import (
            register_pyspark_backends,
        )
        from pandera.config import CONFIG

        register_pyspark_backends(
            use_narwhals_backend=CONFIG.use_narwhals_backend
        )
        return

    if module.startswith("xarray."):
        from pandera.backends.xarray.register import register_xarray_backends

        register_xarray_backends()
        return
