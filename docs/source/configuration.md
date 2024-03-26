```{eval-rst}
.. currentmodule:: pandera
```

(configuration)=

# Configuration

*New in version 0.17.3*
`pandera` provides a global config `~pandera.config.PanderaConfig`.

This configuration can also be set using environment variables. For instance:

```
export PANDERA_VALIDATION_ENABLED=False
export PANDERA_VALIDATION_DEPTH=DATA_ONLY
```

Runtime data validation incurs a performance overhead. To mitigate this, you have
the option to disable validation globally. This can be achieved by setting the
environment variable `PANDERA_VALIDATION_ENABLE=False`. When validation is
disabled, any `validate` call will return `None`.
