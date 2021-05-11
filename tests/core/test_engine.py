from typing import Any, List, Type, Union

import pytest

from pandera.engines.engine import Engine


class BaseDataType:
    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, type(self)):
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


class SimpleDtype(BaseDataType):
    pass


@pytest.fixture
def equivalents() -> List[Any]:
    return [int, "int", 1]


@pytest.fixture
def engine() -> Type[Engine]:
    class FakeEngine(metaclass=Engine, base_datatype=BaseDataType):
        pass

    yield FakeEngine

    del FakeEngine


def test_register_bare_dtype(engine: Type[Engine]):
    """Test that a dtype without  equivalents nor 'from_parametrized_dtype'
    classmethod can be registered.
    """
    with pytest.warns(UserWarning):
        engine.register_dtype(SimpleDtype)


def test_register_equivalents(engine: Type[Engine], equivalents: List[Any]):
    """Test that a dtype with equivalents can be registered."""
    engine.register_dtype(SimpleDtype, equivalents=equivalents)

    for equivalent in equivalents:
        assert engine.dtype(equivalent) == SimpleDtype()

    with pytest.raises(
        TypeError, match="Data type 'foo' not understood by FakeEngine"
    ):
        engine.dtype("foo")


def test_register_from_parametrized_dtype(engine: Type[Engine]):
    """Test that a dtype with from_parametrized_dtype can be registered."""

    @engine.register_dtype
    class Dtype(BaseDataType):
        pass

        @classmethod
        def from_parametrized_dtype(cls, x: int):
            return x

    assert engine.dtype(42) == 42

    with pytest.raises(
        TypeError, match="Data type 'foo' not understood by FakeEngine"
    ):
        engine.dtype("foo")


def test_register_from_parametrized_dtype_union(engine: Type[Engine]):
    """Test that a dtype with from_parametrized_dtype and Union annotation
    can be registered.
    """

    @engine.register_dtype
    class Dtype(BaseDataType):
        pass

        @classmethod
        def from_parametrized_dtype(cls, x: Union[int, str]):
            return x

    assert engine.dtype(42) == 42


def test_register_invalid_from_parametrized_dtype(engine: Type[Engine]):
    """Test that a dtype with invalid from_parametrized_dtype
    cannot be registered.
    """

    with pytest.raises(
        ValueError,
        match="Dtype.from_parametrized_dtype must be a classmethod.",
    ):

        @engine.register_dtype
        class Dtype(BaseDataType):
            pass

            def from_parametrized_dtype(cls, x: int):
                return x


def test_register_dtype_complete(engine: Type[Engine], equivalents: List[Any]):
    """Test that a dtype with equivalents and from_parametrized_dtype
    can be registered.
    """

    @engine.register_dtype(equivalents=equivalents)
    class Dtype(BaseDataType):
        pass

        @classmethod
        def from_parametrized_dtype(cls, x: Union[int, str]):
            return x

    assert engine.dtype(42) == 42
    assert engine.dtype("foo") == "foo"

    for equivalent in equivalents:
        engine.dtype(equivalent) == Dtype()

    with pytest.raises(
        TypeError,
        match="Data type '<class 'str'>' not understood by FakeEngine",
    ):
        engine.dtype(str)


def test_register_dtype_overwrite(
    engine: Type[Engine], equivalents: List[Any]
):
    """Test that register_dtype overwrites existing registrations."""

    @engine.register_dtype(equivalents=["foo"])
    class DtypeA(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: Union[int, str]):
            return DtypeA()

    assert engine.dtype("foo") == DtypeA()
    assert engine.dtype("bar") == DtypeA()
    assert engine.dtype(42) == DtypeA()

    @engine.register_dtype(equivalents=["foo"])
    class DtypeB(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: int):
            return DtypeB()

    assert engine.dtype("foo") == DtypeB()
    assert engine.dtype("bar") == DtypeA()
    assert engine.dtype(42) == DtypeB()


def test_register_base_datatype(engine: Type[Engine]):
    """Test that base datatype cannot be registered."""
    with pytest.raises(
        ValueError,
        match="BaseDataType subclasses cannot be registered with FakeEngine.",
    ):

        @engine.register_dtype(equivalents=[SimpleDtype])
        class Dtype(BaseDataType):
            pass


def test_return_base_dtype(engine: Type[Engine]):
    """Test that Engine.dtype returns back base datatypes."""
    assert engine.dtype(SimpleDtype()) == SimpleDtype()
    assert engine.dtype(SimpleDtype) == SimpleDtype()

    class ParametrizedDtype(BaseDataType):
        def __init__(self, x: int) -> None:
            self.x = x

        def __eq__(self, obj: object) -> bool:
            if not isinstance(obj, ParametrizedDtype):
                return NotImplemented
            return obj.x == self.x

    assert engine.dtype(ParametrizedDtype(1)) == ParametrizedDtype(1)
    with pytest.raises(
        TypeError, match="DataType 'ParametrizedDtype' cannot be instantiated"
    ):
        engine.dtype(ParametrizedDtype)
