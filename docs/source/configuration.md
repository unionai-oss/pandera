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
