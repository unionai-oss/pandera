"""DataFusion backend implementation for schemas and checks.

Validation itself is performed by the narwhals backends
(:mod:`pandera.backends.narwhals`), reached through the ``narwhals-datafusion``
plugin; this package only wires ``datafusion.DataFrame`` into the backend
registry, lazily via
:func:`~pandera.backends.datafusion.register.register_datafusion_backends`.
"""
