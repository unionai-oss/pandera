"""Tests Engine subclassing and registring DataTypes."""
# pylint:disable=redefined-outer-name,unused-argument
# pylint:disable=missing-function-docstring,missing-class-docstring
import re
from typing import Any, Generator, List, Union

import pytest

from pandera.dtypes import DataType
from pandera.engines.engine import Engine


class BaseDataType(DataType):
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
def engine() -> Generator[Engine, None, None]:
    class FakeEngine(  # pylint:disable=too-few-public-methods
        metaclass=Engine, base_pandera_dtypes=BaseDataType
    ):
        pass

    yield FakeEngine

    del FakeEngine


def test_register_equivalents(engine: Engine, equivalents: List[Any]):
    """Test that a dtype with equivalents can be registered."""
    engine.register_dtype(SimpleDtype, equivalents=equivalents)

    for equivalent in equivalents:
        assert engine.dtype(equivalent) == SimpleDtype()

    with pytest.raises(
        TypeError, match="Data type 'foo' not understood by FakeEngine"
    ):
        engine.dtype("foo")


def test_register_from_parametrized_dtype(engine: Engine):
    """Test that a dtype with from_parametrized_dtype can be registered."""

    @engine.register_dtype
    class _Dtype(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: int):
            return x

    assert engine.dtype(42) == 42

    with pytest.raises(
        TypeError, match="Data type 'foo' not understood by FakeEngine"
    ):
        engine.dtype("foo")


def test_register_from_parametrized_dtype_union(engine: Engine):
    """Test that a dtype with from_parametrized_dtype and Union annotation
    can be registered.
    """

    @engine.register_dtype
    class _Dtype(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: Union[int, str]):
            return x

    assert engine.dtype(42) == 42


def test_register_notclassmethod_from_parametrized_dtype(engine: Engine):
    """Test that a dtype with invalid from_parametrized_dtype
    cannot be registered.
    """

    with pytest.raises(
        ValueError,
        match="_InvalidDtype.from_parametrized_dtype must be a classmethod.",
    ):

        @engine.register_dtype
        class _InvalidDtype(BaseDataType):
            def from_parametrized_dtype(  # pylint:disable=no-self-argument,no-self-use
                cls, x: int
            ):
                return x


def test_register_dtype_complete(engine: Engine, equivalents: List[Any]):
    """Test that a dtype with equivalents and from_parametrized_dtype
    can be registered.
    """

    @engine.register_dtype(equivalents=equivalents)
    class _Dtype(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: Union[int, str]):
            return x

    assert engine.dtype(42) == 42
    assert engine.dtype("foo") == "foo"

    for equivalent in equivalents:
        assert engine.dtype(equivalent) == _Dtype()

    with pytest.raises(
        TypeError,
        match="Data type '<class 'str'>' not understood by FakeEngine",
    ):
        engine.dtype(str)


def test_register_dtype_overwrite(engine: Engine):
    """Test that register_dtype overwrites existing registrations."""

    @engine.register_dtype(equivalents=["foo"])
    class _DtypeA(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: Union[int, str]):
            return _DtypeA()

    assert engine.dtype("foo") == _DtypeA()
    assert engine.dtype("bar") == _DtypeA()
    assert engine.dtype(42) == _DtypeA()

    @engine.register_dtype(equivalents=["foo"])
    class _DtypeB(BaseDataType):
        @classmethod
        def from_parametrized_dtype(cls, x: int):
            return _DtypeB()

    assert engine.dtype("foo") == _DtypeB()
    assert engine.dtype("bar") == _DtypeA()
    assert engine.dtype(42) == _DtypeB()


def test_register_base_pandera_dtypes():
    """Test that base datatype cannot be registered."""

    class FakeEngine(  # pylint:disable=too-few-public-methods
        metaclass=Engine, base_pandera_dtypes=(BaseDataType, BaseDataType)
    ):
        pass

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Subclasses of ['tests.core.test_engine.BaseDataType', "
            + "'tests.core.test_engine.BaseDataType'] "
            + "cannot be registered with FakeEngine."
        ),
    ):

        @FakeEngine.register_dtype(equivalents=[SimpleDtype])
        class _Dtype(BaseDataType):
            pass


def test_return_base_dtype(engine: Engine):
    """Test that Engine.dtype returns back base datatypes."""
    assert engine.dtype(SimpleDtype()) == SimpleDtype()
    assert engine.dtype(SimpleDtype) == SimpleDtype()

    class ParametrizedDtypec(BaseDataType):
        def __init__(self, x: int) -> None:
            super().__init__()
            self.x = x

        def __eq__(self, obj: object) -> bool:
            if not isinstance(obj, ParametrizedDtypec):
                return NotImplemented
            return obj.x == self.x

    assert engine.dtype(ParametrizedDtypec(1)) == ParametrizedDtypec(1)
    with pytest.raises(
        TypeError, match="DataType 'ParametrizedDtypec' cannot be instantiated"
    ):
        engine.dtype(ParametrizedDtypec)
