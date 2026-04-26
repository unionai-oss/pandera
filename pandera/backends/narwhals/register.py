"""Narwhals backend registration.

Registration of Narwhals backends for Polars and Ibis frame types is handled by
pandera.backends.polars.register.register_polars_backends() and
pandera.backends.ibis.register.register_ibis_backends() when
PANDERA_USE_NARWHALS_BACKEND=True (or pandera.config.CONFIG.use_narwhals_backend
is True).
"""
