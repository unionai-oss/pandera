"""PyArrow backend implementation for schemas and checks.

Validation itself is performed by the narwhals backends
(:mod:`pandera.backends.narwhals`), which are frame-agnostic; this package
only wires ``pyarrow.Table`` into the backend registry, lazily via
:func:`~pandera.backends.pyarrow.register.register_pyarrow_backends`.
"""
