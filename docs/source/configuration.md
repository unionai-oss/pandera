(configuration)=

# Configuration

*New in version 0.17.3*

`pandera` provides a global config `~pandera.config.PanderaConfig`. The
global configuration is available through `pandera.config.CONFIG`. It can also
be modified with a configuration context `~pandera.config.config_context` and
fetched with `~pandera.config.get_config_context` in custom code.

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

## Narwhals-powered Polars / Ibis backend

*New in version 0.32.0*

Pandera ships an optional
[Narwhals](https://narwhals-dev.github.io/narwhals/)-powered backend that
unifies the Polars and Ibis validation paths. It is **opt-in**; by
default the native Polars and Ibis backends are used. To switch the
`pandera.polars` and `pandera.ibis` integrations onto the Narwhals backend,
install the `narwhals` extra and set:

```bash
export PANDERA_USE_NARWHALS_BACKEND=True
```

Equivalently, set `pandera.config.CONFIG.use_narwhals_backend = True`
before any Polars or Ibis schema is constructed. The backend choice is
locked in at first schema construction (the registration step is
`lru_cache`-d), so toggle this setting at process start. To switch backends
in the same process, call `register_polars_backends.cache_clear()` and/or
`register_ibis_backends.cache_clear()` before re-registering.

If `PANDERA_USE_NARWHALS_BACKEND=True` but `narwhals` is not installed,
schema construction raises an `ImportError` pointing you at
`pandera[narwhals]`. See the
{ref}`Narwhals-powered backends <narwhals-backends>` section of the
supported libraries page for the full feature comparison.
