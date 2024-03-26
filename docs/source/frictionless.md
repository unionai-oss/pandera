```{eval-rst}
.. currentmodule:: pandera
```

(frictionless-integration)=

# Reading Third-Party Schema

*new in 0.7.0*

Pandera now accepts schema from other data validation frameworks. This requires
a pandera installation with the `io` extension; please see the
{ref}`installation<installation>` instructions for more details.

## Frictionless Data Schema

:::{note}
Please see the
[Frictionless schema](https://specs.frictionlessdata.io/table-schema/)
documentation for more information on this standard.
:::

```{eval-rst}
.. autofunction:: pandera.io.from_frictionless_schema
```

under the hood, this uses the {class}`~pandera.io.pandas_io.FrictionlessFieldParser` class
to parse each frictionless field (column):

```{eval-rst}
.. autoclass:: pandera.io.pandas_io.FrictionlessFieldParser
    :members:
```
