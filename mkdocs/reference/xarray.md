[API reference](./index.html) › **Xarray**

---

# Xarray

*New in 0.31.0*

Schemas and components for validating `xarray.DataArray`, `xarray.Dataset`,
and `xarray.DataTree`. Typical imports use `pandera.xarray`; implementations
live under `pandera.api.xarray`. See the [xarray guide](../xarray_guide/index.html)
for usage examples.

The `pandera.xarray` entry point also re-exports
[`Check`](./generated/pandera.api.checks.Check.html), [`Parser`](./generated/pandera.api.parsers.Parser.html),
the [decorators](./../reference/decorators.html) (`check_input`, etc.), and
`pandera.errors` — see [Core](./core.html) for those APIs.

## Schemas

- [`DataArraySchema`](./generated/pandera.api.xarray.container.DataArraySchema.html)
- [`DatasetSchema`](./generated/pandera.api.xarray.container.DatasetSchema.html)
- [`DataTreeSchema`](./generated/pandera.api.xarray.container.DataTreeSchema.html)

## Schema components

- [`DataVar`](./generated/pandera.api.xarray.components.DataVar.html)
- [`Coordinate`](./generated/pandera.api.xarray.components.Coordinate.html)

## Declarative models

- [`DataArrayModel`](./generated/pandera.api.xarray.model.DataArrayModel.html)
- [`DatasetModel`](./generated/pandera.api.xarray.model.DatasetModel.html)
- [`DataTreeModel`](./generated/pandera.api.xarray.model.DataTreeModel.html)
- [`Field`](./generated/pandera.api.xarray.model_components.Field.html)

## Typing

- [`Coordinate`](./generated/pandera.typing.xarray.Coordinate.html)
- [`DataArray`](./generated/pandera.typing.xarray.DataArray.html)
- [`Dataset`](./generated/pandera.typing.xarray.Dataset.html)
- [`DataTree`](./generated/pandera.typing.xarray.DataTree.html)
- [`XarrayAnnotationBase`](./generated/pandera.typing.xarray.XarrayAnnotationBase.html)

## Model configuration

- [`DataArrayConfig`](./generated/pandera.api.xarray.model_config.DataArrayConfig.html)
- [`DatasetConfig`](./generated/pandera.api.xarray.model_config.DatasetConfig.html)
- [`DataTreeConfig`](./generated/pandera.api.xarray.model_config.DataTreeConfig.html)

## Abstract base classes

- [`BaseDataArraySchema`](./generated/pandera.api.xarray.base.BaseDataArraySchema.html)
- [`BaseDatasetSchema`](./generated/pandera.api.xarray.base.BaseDatasetSchema.html)
- [`BaseDataTreeSchema`](./generated/pandera.api.xarray.base.BaseDataTreeSchema.html)

## Check object types

- [`XarrayData`](./generated/pandera.api.xarray.types.XarrayData.html)
- [`XARRAY_CHECK_OBJECT_TYPES`](./generated/pandera.api.xarray.types.XARRAY_CHECK_OBJECT_TYPES.html)
- [`XarrayCheckObjects`](./generated/pandera.api.xarray.types.XarrayCheckObjects.html)

## Configuration

- [`ValidationDepth`](./generated/pandera.config.ValidationDepth.html)
- [`ValidationScope`](./generated/pandera.config.ValidationScope.html)

## Utilities

- [`get_validation_depth`](./generated/pandera.api.xarray.utils.get_validation_depth.html)

## Schema inference

- [`infer_schema`](./generated/pandera.schema_inference.xarray.infer_schema.html)

## IO (YAML / JSON)

- [`to_yaml`](./generated/pandera.io.xarray_io.to_yaml.html)
- [`from_yaml`](./generated/pandera.io.xarray_io.from_yaml.html)
- [`to_json`](./generated/pandera.io.xarray_io.to_json.html)
- [`from_json`](./generated/pandera.io.xarray_io.from_json.html)

## Hypothesis strategies

- [`xarray_dtype_strategy`](./generated/pandera.strategies.xarray_strategies.xarray_dtype_strategy.html)
- [`data_array_strategy`](./generated/pandera.strategies.xarray_strategies.data_array_strategy.html)
- [`dataset_strategy`](./generated/pandera.strategies.xarray_strategies.dataset_strategy.html)
- [`data_array_schema_strategy`](./generated/pandera.strategies.xarray_strategies.data_array_schema_strategy.html)
- [`dataset_schema_strategy`](./generated/pandera.strategies.xarray_strategies.dataset_schema_strategy.html)
