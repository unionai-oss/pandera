# API

The command-line interface (installation, usage, generated help, and Python
entry points) is documented on the dedicated [CLI](../cli.html) page.

## DataFrames

| API | Description |
| --- | --- |
| [Core](./core.html) | The core objects for defining pandera schemas |
| [GeoPandas](./geopandas.html) | `GeoDataFrameSchema` and `GeoDataFrameModel` entry point |
| [Data Types](./dtypes.html) | Data types for type checking and coercion. |
| [DataFrame Models](./dataframe_models.html) | Alternative class-based API for defining types for tabular/array-like data. |
| [Decorators](./decorators.html) | Decorators for integrating pandera schemas with python functions. |
| [Schema Inference](./schema_inference.html) | Bootstrap schemas from real data |
| [IO Utilities](./io.html) | Utility functions for reading/writing schemas |
| [Data Synthesis Strategies](./strategies.html) | Module of functions for generating data from schemas. |
| [Narwhals Backend](./narwhals.html) | Opt-in Narwhals-powered backend unifying the Polars, Ibis, and PySpark SQL validation paths. |
| [Parsers & Derived Columns](./parsers.html) | `Parser`, `ParsedField`/`ParsedColumn` and declaring derived columns |
| [Decision Parsing](./decisions.html) | Parse unstructured data by asking a decision model, plus semantic checks |
| [Extensions](./extensions.html) | Utility functions for extending pandera functionality |
| [Errors](./errors.html) | Pandera-specific exceptions |

## Multi-dimensional arrays

| API | Description |
| --- | --- |
| [Xarray](./xarray.html) | Schemas for labelled N-dimensional `xarray` arrays, datasets, and datatrees |
| [PyTorch](./pytorch.html) | Schemas for `tensordict` `TensorDict` and `tensorclass` objects |

## Configuration

| API | Description |
| --- | --- |
| [`PanderaConfig`](./generated/pandera.config.PanderaConfig.html) | Global configuration (validation enabled, depth, caching) |
| [`ValidationDepth`](./generated/pandera.config.ValidationDepth.html) | Whether to apply checks at schema-level, data-level, or both |
| [`ValidationScope`](./generated/pandera.config.ValidationScope.html) | Indicates whether a check operates at schema or data level |
| [`config_context`](./generated/pandera.config.config_context.html) | Context manager to temporarily override config settings |
| [`get_config_context`](./generated/pandera.config.get_config_context.html) | Get the current configuration context |
| [`get_config_global`](./generated/pandera.config.get_config_global.html) | Get the global configuration |
| [`set_config`](./generated/pandera.config.set_config.html) | Set global configuration options (also available as [`pandera.set_config`](./generated/pandera.set_config.html)) |
| [`reset_config_context`](./generated/pandera.config.reset_config_context.html) | Reset context configuration to the global default |
