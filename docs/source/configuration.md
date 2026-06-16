(configuration)=

# Configuration

*New in version 0.17.3*

`pandera` provides a global config `~pandera.config.PanderaConfig`. The
global configuration is available through `pandera.config.CONFIG`. It can also
be modified with {func}`~pandera.set_config`, a configuration context
`~pandera.config.config_context`, and fetched with
`~pandera.config.get_config_context` in custom code.

This configuration can also be set using environment variables.

## Validation depth

Validation depth determines whether pandera only runs schema-level validations
(column names and datatypes), data-level validations (checks on actual values),
or both:

```
export PANDERA_VALIDATION_ENABLED=False
export PANDERA_VALIDATION_DEPTH=DATA_ONLY  # SCHEMA_AND_DATA, SCHEMA_ONLY, DATA_ONLY
```

## Enabling/disabling validation

Runtime data validation incurs a performance overhead. To mitigate this in the
appropriate contexts, you have the option to disable validation globally.

This can be achieved by setting the environment variable
`PANDERA_VALIDATION_ENABLED=False`. When validation is disabled, any
`validate` call not actually run any validation checks.

## Narwhals-powered backend

*New in version 0.32.0*

Pandera ships an optional
[Narwhals](https://narwhals-dev.github.io/narwhals/)-powered backend that
unifies the Polars, Ibis, and PySpark SQL validation paths. It is **opt-in**; by
default the native backends are used.

Install the `narwhals` extra and enable the backend with either:

```bash
export PANDERA_USE_NARWHALS_BACKEND=True
```

or:

```python
import pandera

pandera.set_config(use_narwhals_backend=True)
```

You can call {func}`~pandera.set_config` before or after importing
``pandera.polars``, ``pandera.ibis``, or ``pandera.pyspark``.

### Lazy registration

Validation backends are **not** registered at import time. Registration happens
lazily the first time a schema is constructed or ``validate()`` is called.
Until then, changing ``CONFIG.use_narwhals_backend`` (via the environment
variable or {func}`~pandera.set_config`) takes effect on the first registration
with no extra steps.

### Runtime re-registration

If {func}`~pandera.set_config` changes ``use_narwhals_backend`` after backends
have already been registered, pandera automatically clears the registration
caches, swaps the backend classes in the registry, and emits a ``UserWarning``.
Existing schema objects keep working — backend lookup happens on each
``validate()`` call.

See {ref}`Backend registration <narwhals-backend-registration>` for worked
examples and the {ref}`Narwhals backend guide <narwhals-backend>` for
installation commands, feature comparison, and PySpark-specific differences.

If `PANDERA_USE_NARWHALS_BACKEND=True` but `narwhals` is not installed,
schema construction raises an `ImportError` pointing you at
`pandera[narwhals]`.
