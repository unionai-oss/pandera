"""Per-backend fixtures for narwhals backend tests.

Tests in this directory run once per backend using native frames.
Narwhals is an implementation detail — never pass nw-wrapped frames here.

Adding a new backend: subclass BackendFixture, implement its methods,
and add a pytest.param entry to BACKENDS.
"""

import pytest


class BackendFixture:
    name: str
    DataFrameSchema: type
    Column: type

    def make_frame(self, data: dict):
        raise NotImplementedError


class PolarsBackend(BackendFixture):
    name = "polars"

    @property
    def DataFrameSchema(self):
        import pandera.polars as pa_pl

        return pa_pl.DataFrameSchema

    @property
    def Column(self):
        import pandera.polars as pa_pl

        return pa_pl.Column

    def make_frame(self, data: dict):
        import polars as pl

        return pl.DataFrame(data)


class IbisBackend(BackendFixture):
    name = "ibis"

    @property
    def DataFrameSchema(self):
        import pandera.ibis as pa_ibis

        return pa_ibis.DataFrameSchema

    @property
    def Column(self):
        import pandera.ibis as pa_ibis

        return pa_ibis.Column

    def make_frame(self, data: dict):
        import ibis

        return ibis.memtable(data)


BACKENDS = [
    pytest.param(PolarsBackend(), id="polars", marks=pytest.mark.polars),
    pytest.param(IbisBackend(), id="ibis", marks=pytest.mark.ibis),
]


@pytest.fixture(params=BACKENDS)
def backend(request) -> BackendFixture:
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
