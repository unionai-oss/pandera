"""Lazy registration of check backends."""

from __future__ import annotations


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

    from pandera.api.pandas.types import get_backend_types_from_mro

    if get_backend_types_from_mro(check_obj_cls) is not None:
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

    if module.startswith("pyspark"):
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
