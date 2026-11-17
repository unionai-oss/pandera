# Narwhals Backend

*new in 0.32.0*

Opt-in [Narwhals](https://narwhals-dev.github.io/narwhals/)-powered
validation backend that powers the Polars, Ibis, PySpark SQL, and pandas
integrations behind a single unified code path. Requires the `narwhals`
extra. Enable with `PANDERA_USE_NARWHALS_BACKEND=True` or
[`pandera.set_config`](./generated/pandera.set_config.html). Backends register
lazily on first schema use; changing the flag at runtime triggers automatic
re-registration. See [Backend registration](../narwhals_backend.html) and
[Narwhals-powered backends](../narwhals_backend.html) for the user-facing guide.

## Data Objects

- [`NarwhalsData`](./generated/pandera.api.narwhals.types.NarwhalsData.html)
- [`NarwhalsCheckResult`](./generated/pandera.api.narwhals.types.NarwhalsCheckResult.html)

## Backends

- [`NarwhalsSchemaBackend`](./generated/pandera.backends.narwhals.base.NarwhalsSchemaBackend.html)
- [`DataFrameSchemaBackend`](./generated/pandera.backends.narwhals.container.DataFrameSchemaBackend.html)
- [`ColumnBackend`](./generated/pandera.backends.narwhals.components.ColumnBackend.html)
- [`NarwhalsCheckBackend`](./generated/pandera.backends.narwhals.checks.NarwhalsCheckBackend.html)

## Narwhals Dtypes

- [`DataType`](./generated/pandera.engines.narwhals_engine.DataType.html)
- [`Int8`](./generated/pandera.engines.narwhals_engine.Int8.html)
- [`Int16`](./generated/pandera.engines.narwhals_engine.Int16.html)
- [`Int32`](./generated/pandera.engines.narwhals_engine.Int32.html)
- [`Int64`](./generated/pandera.engines.narwhals_engine.Int64.html)
- [`UInt8`](./generated/pandera.engines.narwhals_engine.UInt8.html)
- [`UInt16`](./generated/pandera.engines.narwhals_engine.UInt16.html)
- [`UInt32`](./generated/pandera.engines.narwhals_engine.UInt32.html)
- [`UInt64`](./generated/pandera.engines.narwhals_engine.UInt64.html)
- [`Float32`](./generated/pandera.engines.narwhals_engine.Float32.html)
- [`Float64`](./generated/pandera.engines.narwhals_engine.Float64.html)
- [`String`](./generated/pandera.engines.narwhals_engine.String.html)
- [`Bool`](./generated/pandera.engines.narwhals_engine.Bool.html)
- [`Date`](./generated/pandera.engines.narwhals_engine.Date.html)
- [`DateTime`](./generated/pandera.engines.narwhals_engine.DateTime.html)
- [`Duration`](./generated/pandera.engines.narwhals_engine.Duration.html)
- [`Categorical`](./generated/pandera.engines.narwhals_engine.Categorical.html)
- [`List`](./generated/pandera.engines.narwhals_engine.List.html)
- [`Struct`](./generated/pandera.engines.narwhals_engine.Struct.html)

## Engine

- [`Engine`](./generated/pandera.engines.narwhals_engine.Engine.html)

## Built-in Check Expressions

- [`equal_to`](./generated/pandera.backends.narwhals.builtin_checks.equal_to.html)
- [`not_equal_to`](./generated/pandera.backends.narwhals.builtin_checks.not_equal_to.html)
- [`greater_than`](./generated/pandera.backends.narwhals.builtin_checks.greater_than.html)
- [`greater_than_or_equal_to`](./generated/pandera.backends.narwhals.builtin_checks.greater_than_or_equal_to.html)
- [`less_than`](./generated/pandera.backends.narwhals.builtin_checks.less_than.html)
- [`less_than_or_equal_to`](./generated/pandera.backends.narwhals.builtin_checks.less_than_or_equal_to.html)
- [`in_range`](./generated/pandera.backends.narwhals.builtin_checks.in_range.html)
- [`isin`](./generated/pandera.backends.narwhals.builtin_checks.isin.html)
- [`notin`](./generated/pandera.backends.narwhals.builtin_checks.notin.html)
- [`str_matches`](./generated/pandera.backends.narwhals.builtin_checks.str_matches.html)
- [`str_contains`](./generated/pandera.backends.narwhals.builtin_checks.str_contains.html)
- [`str_startswith`](./generated/pandera.backends.narwhals.builtin_checks.str_startswith.html)
- [`str_endswith`](./generated/pandera.backends.narwhals.builtin_checks.str_endswith.html)
- [`str_length`](./generated/pandera.backends.narwhals.builtin_checks.str_length.html)
