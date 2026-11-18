[API reference](./index.html) › **PyTorch**

---

# PyTorch

*New in 0.32.0*

Schemas and components for validating `tensordict.TensorDict` and
`tensordict.tensorclass`. Typical imports use `pandera.tensordict`;
implementations live under `pandera.api.tensordict`. See the
[pytorch guide](../pytorch_guide/index.html) for usage examples.

The `pandera.tensordict` entry point also re-exports
[`Check`](./generated/pandera.api.checks.Check.html) and `pandera.errors` —
see [Core](./core.html) for those APIs.

## Schemas

- [`TensorDictSchema`](./generated/pandera.api.tensordict.container.TensorDictSchema.html)

## Schema components

- [`Tensor`](./generated/pandera.api.tensordict.components.Tensor.html)

## Declarative models

- [`TensorDictModel`](./generated/pandera.api.tensordict.model.TensorDictModel.html)
- [`Field`](./generated/pandera.api.tensordict.model_components.Field.html)

## Configuration

- [`ValidationDepth`](./generated/pandera.config.ValidationDepth.html)
- [`ValidationScope`](./generated/pandera.config.ValidationScope.html)
