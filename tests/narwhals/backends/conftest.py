"""Per-backend fixtures for narwhals backend tests.

Tests in this directory run once per backend using native frames.
Narwhals is an implementation detail — never pass nw-wrapped frames here.

Adding a new backend: subclass _BackendFixture, implement its methods,
and add a pytest.param entry to _BACKENDS.
"""

import polars as pl
import pytest

import pandera.ibis as pa_ibis
import pandera.polars as pa_pl


class _BackendFixture:
    name: str
    DataFrameSchema: type
    Column: type

    def make_frame(self, data: dict):
        raise NotImplementedError


class _PolarsBackend(_BackendFixture):
    name = "polars"
    DataFrameSchema = pa_pl.DataFrameSchema
    Column = pa_pl.Column

    def make_frame(self, data: dict):
        return pl.DataFrame(data)


class _IbisBackend(_BackendFixture):
    name = "ibis"
    DataFrameSchema = pa_ibis.DataFrameSchema
    Column = pa_ibis.Column

    def make_frame(self, data: dict):
        import ibis

        return ibis.memtable(data)


_BACKENDS = [
    pytest.param(_PolarsBackend(), id="polars"),
    pytest.param(_IbisBackend(), id="ibis"),
]


@pytest.fixture(params=_BACKENDS)
def backend(request) -> _BackendFixture:
    return request.param


@pytest.fixture
def DataFrameSchema(backend):
    return backend.DataFrameSchema


@pytest.fixture
def Column(backend):
    return backend.Column


@pytest.fixture
def frame(backend):
    """A simple valid frame for the current backend."""
    return backend.make_frame({"x": [1, 2, 3], "y": [10, 20, 30]})
