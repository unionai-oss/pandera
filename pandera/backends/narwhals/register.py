"""Narwhals backend registration utilities.

Registration of Narwhals backends for Polars, Ibis, and PySpark frame types is
handled by ``register_polars_backends()``, ``register_ibis_backends()``, and
``register_pyspark_backends()`` when ``PANDERA_USE_NARWHALS_BACKEND=True`` (or
``pandera.config.CONFIG.use_narwhals_backend`` is ``True``).
"""

from __future__ import annotations

import warnings
from typing import Any


def _pop_registry_key(
    schema_cls: type[Any],
    frame_type: type[Any],
) -> None:
    schema_cls.BACKEND_REGISTRY.pop((schema_cls, frame_type), None)  # type: ignore[attr-defined]


def clear_narwhals_compatible_backend_registry() -> None:
    """Remove registry entries for narwhals-compatible backends."""
    from pandera.api.checks import Check

    try:
        import polars as pl

        from pandera.api.polars.components import Column as PolarsColumn
        from pandera.api.polars.container import (
            DataFrameSchema as PolarsDataFrameSchema,
        )

        polars_keys = [
            (PolarsDataFrameSchema, pl.LazyFrame),
            (PolarsDataFrameSchema, pl.DataFrame),
            (PolarsColumn, pl.LazyFrame),
            (Check, pl.LazyFrame),
        ]
        try:
            import narwhals.stable.v1 as nw

            polars_keys.extend(
                [
                    (Check, nw.LazyFrame),
                    (Check, nw.DataFrame),
                ]
            )
        except ImportError:
            pass

        for schema_cls, frame_type in polars_keys:
            _pop_registry_key(schema_cls, frame_type)
    except ImportError:
        pass

    try:
        import ibis

        from pandera.api.ibis.components import Column as IbisColumn
        from pandera.api.ibis.container import (
            DataFrameSchema as IbisDataFrameSchema,
        )

        ibis_keys = [
            (IbisDataFrameSchema, ibis.Table),
            (IbisColumn, ibis.Table),
            (Check, ibis.Table),
            (Check, ibis.Column),
        ]
        try:
            import narwhals.stable.v1 as nw

            ibis_keys.append((Check, nw.LazyFrame))
        except ImportError:
            pass

        for schema_cls, frame_type in ibis_keys:
            _pop_registry_key(schema_cls, frame_type)
    except ImportError:
        pass

    try:
        import pyspark.sql as pyspark_sql

        from pandera.api.dataframe.components import ComponentSchema
        from pandera.api.pyspark.components import Column as PySparkColumn
        from pandera.api.pyspark.container import (
            DataFrameSchema as PySparkDataFrameSchema,
        )

        pyspark_keys = [
            (PySparkDataFrameSchema, pyspark_sql.DataFrame),
            (PySparkColumn, pyspark_sql.DataFrame),
            (Check, pyspark_sql.DataFrame),
            (ComponentSchema, pyspark_sql.DataFrame),
        ]

        try:
            import narwhals.stable.v1 as nw

            pyspark_keys.append((Check, nw.LazyFrame))
        except ImportError:
            pass

        try:
            from pyspark.sql.connect import dataframe as pyspark_connect

            pyspark_keys.extend(
                [
                    (
                        PySparkDataFrameSchema,
                        pyspark_connect.DataFrame,
                    ),
                    (PySparkColumn, pyspark_connect.DataFrame),
                    (Check, pyspark_connect.DataFrame),
                    (ComponentSchema, pyspark_connect.DataFrame),
                ]
            )
        except ImportError:
            pass

        for schema_cls, frame_type in pyspark_keys:
            _pop_registry_key(schema_cls, frame_type)
    except ImportError:
        pass


def _narwhals_compatible_registration_state() -> dict[str, bool]:
    state = {
        "polars": False,
        "ibis": False,
        "pyspark": False,
    }

    try:
        from pandera.backends.polars.register import register_polars_backends

        state["polars"] = register_polars_backends.cache_info().currsize > 0
    except ImportError:
        pass

    try:
        from pandera.backends.ibis.register import register_ibis_backends

        state["ibis"] = register_ibis_backends.cache_info().currsize > 0
    except ImportError:
        pass

    try:
        from pandera.backends.pyspark.register import register_pyspark_backends

        state["pyspark"] = register_pyspark_backends.cache_info().currsize > 0
    except ImportError:
        pass

    return state


def _get_register_functions() -> dict[str, Any]:
    register_functions: dict[str, Any] = {}

    try:
        from pandera.backends.polars.register import register_polars_backends

        register_functions["polars"] = register_polars_backends
    except ImportError:
        pass

    try:
        from pandera.backends.ibis.register import register_ibis_backends

        register_functions["ibis"] = register_ibis_backends
    except ImportError:
        pass

    try:
        from pandera.backends.pyspark.register import register_pyspark_backends

        register_functions["pyspark"] = register_pyspark_backends
    except ImportError:
        pass

    return register_functions


def reregister_narwhals_compatible_backends(
    *,
    use_narwhals_backend: bool,
) -> None:
    """Re-register narwhals-compatible backends after toggling the config flag."""
    register_functions = _get_register_functions()
    previously_registered = _narwhals_compatible_registration_state()

    for register_fn in register_functions.values():
        register_fn.cache_clear()

    if not any(previously_registered.values()):
        return

    clear_narwhals_compatible_backend_registry()

    warnings.warn(
        "Re-registered pandera backends after use_narwhals_backend changed. "
        "Existing schema objects continue to work; backend classes were "
        "swapped in the registry.",
        UserWarning,
        stacklevel=3,
    )

    for name, register_fn in register_functions.items():
        if previously_registered.get(name):
            register_fn(use_narwhals_backend=use_narwhals_backend)
