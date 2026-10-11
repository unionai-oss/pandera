"""Tests for the phase-2 TensorDict features:

1. Nested (tuple) keys in schemas and models.
2. Check errors pointing at the failing element indices.
3. Custom checks in ``TensorDictModel`` fields.
4. ``@check_types`` decorator integration.
"""

import pytest

try:
    import torch
    from tensordict import TensorDict, tensorclass

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TensorDict = None
    tensorclass = None
    TORCH_AVAILABLE = False

torch_condition = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch/tensordict not installed"
)


@torch_condition
class TestNestedKeys:
    """Nested (tuple) key support in TensorDictSchema."""

    def test_nested_keys_schema_validation(self):
        import pandera.tensordict as pa

        schema = pa.TensorDictSchema(
            keys={
                ("next", "observation"): pa.Tensor(
                    dtype=torch.float32, shape=(None, 4, 3)
                ),
                ("next", "reward"): pa.Tensor(
                    dtype=torch.float32, shape=(None,)
                ),
                "action": pa.Tensor(dtype=torch.int64, shape=(None,)),
            },
            batch_size=(2,),
        )

        td = TensorDict(
            {
                "action": torch.randint(0, 3, (2,)),
                "next": TensorDict(
                    {
                        "observation": torch.randn(2, 4, 3),
                        "reward": torch.randn(2),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )
        result = schema.validate(td)
        assert isinstance(result, TensorDict)

    def test_nested_keys_missing_raises(self):
        import pandera.errors as errors
        import pandera.tensordict as pa

        schema = pa.TensorDictSchema(
            keys={
                ("next", "observation"): pa.Tensor(dtype=torch.float32),
                ("next", "reward"): pa.Tensor(dtype=torch.float32),
            },
            batch_size=(2,),
        )

        td = TensorDict(
            {
                "next": TensorDict(
                    {"observation": torch.randn(2, 4, 3)}, batch_size=[2]
                ),
            },
            batch_size=[2],
        )
        with pytest.raises(errors.SchemaError, match="reward"):
            schema.validate(td)

    def test_nested_key_wrong_dtype(self):
        import pandera.errors as errors
        import pandera.tensordict as pa

        schema = pa.TensorDictSchema(
            keys={("next", "observation"): pa.Tensor(dtype=torch.float32)},
            batch_size=(2,),
        )
        td = TensorDict(
            {
                "next": TensorDict(
                    {"observation": torch.randn(2, 4, 3).to(torch.float64)},
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )
        with pytest.raises(errors.SchemaError, match="dtype"):
            schema.validate(td)

    def test_nested_keys_model_alias(self):
        import pandera.tensordict as pa

        class Model(pa.TensorDictModel):
            observation: torch.float32 = pa.Field(
                shape=(None, 4, 3), alias=("next", "observation")
            )
            reward: torch.float32 = pa.Field(
                shape=(None,), alias=("next", "reward")
            )
            action: torch.int64 = pa.Field(shape=(None,))

            class Config:
                batch_size = (2,)

        schema = Model.to_schema()
        assert ("next", "observation") in schema.keys
        assert ("next", "reward") in schema.keys

        td = TensorDict(
            {
                "action": torch.randint(0, 3, (2,)),
                "next": TensorDict(
                    {
                        "observation": torch.randn(2, 4, 3),
                        "reward": torch.randn(2),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )
        Model.validate(td)

    def test_nested_key_value_check_failure(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        schema = pa.TensorDictSchema(
            keys={
                ("next", "observation"): pa.Tensor(
                    dtype=torch.float32,
                    shape=(None,),
                    checks=Check.gt(0.0),
                ),
            },
            batch_size=(3,),
        )
        td = TensorDict(
            {
                "next": TensorDict(
                    {
                        "observation": torch.tensor([1.0, -2.0, 3.0]),
                    },
                    batch_size=[3],
                ),
            },
            batch_size=[3],
        )
        with pytest.raises(errors.SchemaError) as exc_info:
            schema.validate(td)
        failure_cases = exc_info.value.failure_cases
        assert failure_cases is not None
        assert failure_cases["index"] == [1]
        assert "next" in str(exc_info.value)


@torch_condition
class TestCheckFailureCases:
    """Element-level failure-case reporting for tensor checks."""

    def test_1d_failure_indices(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        schema = pa.TensorDictSchema(
            keys={
                "values": pa.Tensor(
                    dtype=torch.float32,
                    shape=(None,),
                    checks=Check.gt(0.0),
                ),
            },
            batch_size=(5,),
        )
        td = TensorDict(
            {"values": torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])},
            batch_size=[5],
        )
        with pytest.raises(errors.SchemaError) as exc_info:
            schema.validate(td)
        failure_cases = exc_info.value.failure_cases
        assert failure_cases["index"] == [1, 3]
        assert failure_cases["failure_case"] == [-2.0, -4.0]
        assert "index [1, 3]" in str(exc_info.value)

    def test_2d_failure_index_tuples(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        schema = pa.TensorDictSchema(
            keys={
                "mat": pa.Tensor(
                    dtype=torch.float32,
                    shape=(2, 3),
                    checks=Check.gt(0.0),
                ),
            },
            batch_size=(2,),
        )
        data = torch.tensor([[1.0, -1.0, 2.0], [3.0, 4.0, -5.0]])
        td = TensorDict({"mat": data}, batch_size=[2])
        with pytest.raises(errors.SchemaError) as exc_info:
            schema.validate(td)
        failure_cases = exc_info.value.failure_cases
        assert (0, 1) in failure_cases["index"]
        assert (1, 2) in failure_cases["index"]

    def test_failure_cases_lazy_validation(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        schema = pa.TensorDictSchema(
            keys={
                "a": pa.Tensor(
                    dtype=torch.float32,
                    shape=(None,),
                    checks=Check.gt(0.0),
                ),
                "b": pa.Tensor(
                    dtype=torch.float32,
                    shape=(None,),
                    checks=Check.gt(0.0),
                ),
            },
            batch_size=(3,),
        )
        td = TensorDict(
            {
                "a": torch.tensor([1.0, -2.0, -3.0]),
                "b": torch.tensor([-1.0, 2.0, 3.0]),
            },
            batch_size=[3],
        )
        with pytest.raises(errors.SchemaErrors) as exc_info:
            schema.validate(td, lazy=True)
        check_errors = [
            e for e in exc_info.value.schema_errors if e.check is not None
        ]
        assert len(check_errors) == 2
        # Each check error carries its own failure cases (indices).
        for e in check_errors:
            assert e.failure_cases["index"]


@torch_condition
class TestCustomChecksInModelFields:
    """One-off custom checks attached to TensorDictModel fields."""

    def test_model_field_with_explicit_check(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        def is_one_hot(t):
            return (t.sum(dim=-1) == 1.0) & ((t == 0) | (t == 1)).all(dim=-1)

        class Model(pa.TensorDictModel):
            probs: torch.float32 = pa.Field(
                shape=(None, 3),
                checks=Check(is_one_hot, name="one_hot", element_wise=False),
            )

            class Config:
                batch_size = (3,)

        schema = Model.to_schema()
        assert len(schema.keys["probs"].checks) == 1

        valid = TensorDict(
            {
                "probs": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            },
            batch_size=[3],
        )
        Model.validate(valid)

        invalid = TensorDict(
            {
                "probs": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
                )
            },
            batch_size=[3],
        )
        with pytest.raises(errors.SchemaError) as exc_info:
            Model.validate(invalid)
        # The error should point at the failing batch index.
        assert exc_info.value.failure_cases["index"] == [1]

    def test_model_field_with_combined_checks(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        class Model(pa.TensorDictModel):
            bottles: torch.int64 = pa.Field(
                shape=(None,),
                gt=0,  # intrinsic check
                checks=Check.isin([1, 2, 3]),  # explicit Check
            )

            class Config:
                batch_size = (3,)

        schema = Model.to_schema()
        assert len(schema.keys["bottles"].checks) == 2

        with pytest.raises(errors.SchemaError):
            Model.validate(
                TensorDict(
                    {"bottles": torch.tensor([1, 4, 2])}, batch_size=[3]
                )
            )


@torch_condition
class TestCheckTypesIntegration:
    """@check_types decorator integration for TensorDict."""

    def test_check_types_validates_args_and_returns(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import check_types
        from pandera.typing.tensordict import TensorDict as TDT

        class Obs(pa.TensorDictModel):
            observation: torch.float32 = pa.Field(shape=(None, 4, 3))
            action: torch.int64 = pa.Field(shape=(None,))

            class Config:
                batch_size = (2,)

        @check_types
        def fn(td: TDT[Obs]) -> TDT[Obs]:
            return td

        valid = TensorDict(
            {
                "observation": torch.randn(2, 4, 3),
                "action": torch.randint(0, 3, (2,)),
            },
            batch_size=[2],
        )
        out = fn(valid)
        assert isinstance(out, TensorDict)

        invalid = TensorDict(
            {
                "observation": torch.randn(2, 4, 3).to(torch.float64),
                "action": torch.randint(0, 3, (2,)),
            },
            batch_size=[2],
        )
        with pytest.raises((errors.SchemaError, errors.SchemaErrors)):
            fn(invalid)

    def test_check_types_with_optional(self):
        import pandera.tensordict as pa
        from pandera import check_types
        from pandera.typing.tensordict import TensorDict as TDT

        class Obs(pa.TensorDictModel):
            observation: torch.float32 = pa.Field(shape=(None, 4))

            class Config:
                batch_size = (2,)

        @check_types
        def fn(td: TDT[Obs] | None = None):
            return td

        assert fn(None) is None
        assert (
            fn(TensorDict({"observation": torch.randn(2, 4)}, batch_size=[2]))
            is not None
        )


@torch_condition
class TestNoNanNoInfInFields:
    """no_nan / no_inf field options are wired to checks."""

    def test_no_nan(self):
        import pandera.errors as errors
        import pandera.tensordict as pa

        class M(pa.TensorDictModel):
            values: torch.float32 = pa.Field(shape=(None,), no_nan=True)

            class Config:
                batch_size = (3,)

        schema = M.to_schema()
        assert any(c.name == "no_nan" for c in schema.keys["values"].checks)

        M.validate(
            TensorDict(
                {"values": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3]
            )
        )

        with pytest.raises(errors.SchemaError) as exc_info:
            M.validate(
                TensorDict(
                    {"values": torch.tensor([1.0, float("nan"), 3.0])},
                    batch_size=[3],
                )
            )
        assert exc_info.value.failure_cases["index"] == [1]

    def test_no_inf(self):
        import pandera.errors as errors
        import pandera.tensordict as pa

        class M(pa.TensorDictModel):
            values: torch.float32 = pa.Field(shape=(None,), no_inf=True)

            class Config:
                batch_size = (3,)

        M.validate(
            TensorDict(
                {"values": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3]
            )
        )

        with pytest.raises(errors.SchemaError):
            M.validate(
                TensorDict(
                    {"values": torch.tensor([1.0, float("inf"), 3.0])},
                    batch_size=[3],
                )
            )


@torch_condition
class TestCheckFunctionErrors:
    """A raising check function surfaces its underlying message."""

    def test_raise_message_is_surfaced(self):
        import pandera.errors as errors
        import pandera.tensordict as pa
        from pandera import Check

        def boom(_):
            raise RuntimeError("boom-check-failed")

        schema = pa.TensorDictSchema(
            keys={"v": pa.Tensor(checks=Check(boom))}, batch_size=(2,)
        )
        with pytest.raises(errors.SchemaError, match="boom-check-failed"):
            schema.validate(TensorDict({"v": torch.randn(2)}, batch_size=[2]))
