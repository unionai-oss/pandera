"""Test pickling behavior of SchemaError and SchemaErrors.

See issue #713:
In a multiprocessing or concurrent.futures.ProcessPoolExecutor
situation if a subprocess does not handle the SchemaError itself, it is pickled
and piped back to the main process. This requires picklability, and limits the
pickled size to 2 GiB.
The Exception contains full data and schema. Most Check objects are not picklable.
DataFrames may be large. The signature of SchemaError needs special unpickling
behavior.
"""
import multiprocessing
import pickle
from typing import NoReturn, cast

import numpy as np
import pandas as pd
import pytest

from pandera import Check, Column, DataFrameSchema
from pandera.engines import pandas_engine
from pandera.errors import (
    ParserError,
    ReducedPickleExceptionBase,
    SchemaError,
    SchemaErrors,
)


class MyError(ReducedPickleExceptionBase):
    """Generic implementation of `ReducedPickleExceptionBase`."""

    TO_STRING_KEYS = ["foo_a", "foo_c"]

    def __init__(self, foo_a, msg, foo_b):
        super().__init__(msg)
        self.foo_a = foo_a
        self.foo_b = foo_b
        self.foo_c = 1337


@pytest.fixture(name="reduced_pickle_exception")
def fixture_reduced_pickle_exception() -> MyError:
    """Fixture for MyError instance."""
    return MyError(
        foo_a=lambda x: x + 1,
        msg="error message",
        foo_b=42,
    )


class TestReducedPickleException:
    """Test pickling behavior of `ReducedPickleExceptionBase`.

    Uses `MyError` implementation.
    """

    @staticmethod
    def test_pickle(reduced_pickle_exception: MyError) -> None:
        """Simple test for non-empty pickled object"""
        pickled = pickle.dumps(reduced_pickle_exception)
        assert pickled

    def test_unpickle(self, reduced_pickle_exception: MyError) -> None:
        """Test for expected unpickling behavior.

        Expects a warning.
        """
        pickled = pickle.dumps(reduced_pickle_exception)
        with pytest.warns(
            UserWarning,
            match="Pickling MyError does not preserve state: Attributes "
            r"\['foo_a', 'foo_c'\] become string representations.",
        ):
            unpickled = cast(MyError, pickle.loads(pickled))
        self.validate_unpickled(unpickled)

    @staticmethod
    def validate_unpickled(exc_unpickled: MyError) -> None:
        """Validate unpickled exception."""
        assert str(exc_unpickled) == "error message"
        assert isinstance(exc_unpickled.foo_a, str)
        assert "<locals>.<lambda>" in exc_unpickled.foo_a
        assert exc_unpickled.foo_b == 42
        assert exc_unpickled.foo_c == "1337"

    @staticmethod
    def raise_my_error() -> NoReturn:
        """Throw MyError instance.

        Intended for subprocesses.
        """
        raise MyError(
            foo_a=lambda x: x + 1,
            msg="error message",
            foo_b=42,
        )

    def test_subprocess(self):
        """Test exception propagation from subprocess."""
        with multiprocessing.Pool(1) as pool:
            try:
                with pytest.warns(
                    UserWarning,
                    match="Pickling MyError does not preserve state: Attributes "
                    r"\['foo_a', 'foo_c'\] become string representations.",
                ):
                    pool.apply(self.raise_my_error, args=())
            except MyError as exc:
                self.validate_unpickled(exc)
            else:
                pytest.fail("MyError not raised")


@pytest.fixture(name="int_dataframe")
def fixture_int_dataframe() -> pd.DataFrame:
    """Fixture for simple DataFrame with one negative value."""
    return pd.DataFrame({"a": [-1, 0, 1]})


def _multi_check_schema() -> DataFrameSchema:
    """Schema with multiple positivity checks on column `a`"""
    return DataFrameSchema(
        {
            "a": Column(
                int,
                [
                    Check.isin([0, 1]),
                    Check(lambda x: x >= 0),
                ],
            ),
        }
    )


@pytest.fixture(name="multi_check_schema")
def fixture_multi_check_schema() -> DataFrameSchema:
    """Schema with multiple positivity checks on column `a`"""
    return _multi_check_schema()


@pytest.mark.filterwarnings("ignore:Pickling SchemaError")
class TestSchemaError:
    """Tests pickling behavior of errors.SchemaError."""

    @staticmethod
    @pytest.mark.parametrize(
        "check_obj", [Check.isin([0, 1]), Check(lambda x: x >= 0)]
    )
    def test_pickling(int_dataframe: pd.DataFrame, check_obj: Check):
        """Test for a non-empty pickled object."""
        schema = DataFrameSchema({"a": Column(int, check_obj)})
        try:
            # fails for element -1
            schema.validate(int_dataframe)
        except SchemaError as exc:
            # must be non-empty byte-array
            assert pickle.dumps(exc)
        else:
            pytest.fail("SchemaError not raised")

    @pytest.mark.parametrize("n_tile", [1, 10000])
    def test_unpickling(self, int_dataframe: pd.DataFrame, n_tile: int):
        """Tests content validity of unpickled SchemaError."""
        df = pd.DataFrame(
            {"a": np.tile(int_dataframe["a"].to_numpy(), n_tile)}
        )
        schema = DataFrameSchema({"a": Column(int, Check.isin([0, 1]))})
        loaded = None
        try:
            # fails for element -1
            schema.validate(df)
        except SchemaError as exc:
            loaded = cast(SchemaError, pickle.loads(pickle.dumps(exc)))
        else:
            pytest.fail("SchemaError not raised")
        assert loaded is not None
        self._validate_error(df, n_tile, loaded)

    @staticmethod
    def _validate_error(df: pd.DataFrame, n_tile: int, exc: SchemaError):
        """General validation of Exception content."""
        assert exc is not None
        assert (
            "Schema Column(name=a, type=DataType(int64))> "
            "failed element-wise validator 0" in str(exc)
        )
        assert exc.schema == "<Schema Column(name=a, type=DataType(int64))>"
        assert exc.data == str(df)
        # `failure_cases` is limited to 10 by `n_failure_cases` of `Check`
        assert exc.failure_cases == str(
            pd.DataFrame(
                {
                    "index": np.arange(n_tile) * 3,
                    "failure_case": np.full(n_tile, fill_value=-1, dtype=int),
                }
            )
        )
        assert exc.check == "<Check isin: isin({0, 1})>"
        assert exc.check_index == 0
        assert exc.check_output == str(
            pd.Series(np.tile([False, True, True], n_tile), name="a")
        )


@pytest.mark.filterwarnings("ignore:Pickling SchemaError")
class TestSchemaErrors:
    """Tests pickling behavior of errors.SchemaErrors."""

    @staticmethod
    @pytest.mark.parametrize(
        "schema",
        [
            DataFrameSchema(
                {
                    "a": Column(
                        int,
                        [
                            Check.isin([0, 1]),
                            Check(lambda x: x >= 0),
                        ],
                    ),
                }
            ),
            DataFrameSchema(
                {
                    "a": Column(int, Check.isin([0, 1])),
                }
            ),
        ],
    )
    def test_pickling(int_dataframe: pd.DataFrame, schema: DataFrameSchema):
        """Test for a non-empty pickled object."""
        try:
            schema.validate(int_dataframe, lazy=True)
        except SchemaErrors as exc:
            # expect non-empty bytes
            assert pickle.dumps(exc)
        else:
            pytest.fail("SchemaErrors not raised")

    def test_unpickling(
        self, int_dataframe: pd.DataFrame, multi_check_schema: DataFrameSchema
    ):
        """Tests content validity of unpickled SchemaErrors."""
        try:
            multi_check_schema.validate(int_dataframe, lazy=True)
        except SchemaErrors as exc:
            loaded = pickle.loads(pickle.dumps(exc))
            assert loaded is not None
            self._compare_exception_with_unpickled(exc, loaded)
        else:
            pytest.fail("SchemaErrors not raised")

    @staticmethod
    def _compare_exception_with_unpickled(
        exc_native: SchemaErrors, exc_unpickled: SchemaErrors
    ):
        """Compare content of native SchemaErrors with unpickled one."""
        assert isinstance(exc_native, SchemaErrors)
        assert isinstance(exc_unpickled, SchemaErrors)
        # compare message
        assert str(exc_unpickled) == str(exc_native)
        # compare schema_errors as string, as it is a nested container with
        # elements that compare by identity
        assert str(exc_unpickled.schema_errors) == str(
            exc_native.schema_errors
        )
        assert exc_unpickled.error_counts == exc_native.error_counts
        assert exc_unpickled.failure_cases == str(exc_native.failure_cases)
        assert exc_unpickled.data == str(exc_native.data)


@pytest.mark.filterwarnings("ignore:Pickling ParserError")
def test_pickling_parser_error():
    """Test pickling behavior of ParserError."""
    try:
        pandas_engine.Engine.dtype(int).try_coerce(pd.Series(["a", 0, "b"]))
    except ParserError as exc:
        pickled = pickle.dumps(exc)
        # must be non-empty byte-array
        assert pickled
        unpickled = pickle.loads(pickled)
        assert str(unpickled) == str(exc)
        assert unpickled.failure_cases == str(exc.failure_cases)
    else:
        pytest.fail("ParserError not raised")
