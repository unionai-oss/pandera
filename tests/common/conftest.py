"""Per-backend fixtures for common cross-backend tests.

Tests in this directory run once per backend using native frames.
Pandera dispatches to whichever backend is active (native or narwhals).

Adding a new backend: subclass BackendFixture, implement its methods,
and add a pytest.param entry to BACKENDS.
"""

import pytest

from pandera import dtypes
from pandera.config import CONFIG, ValidationDepth, reset_config_context


@pytest.fixture(scope="function", autouse=True)
def validation_depth_schema_and_data():
    depth = CONFIG.validation_depth
    CONFIG.validation_depth = ValidationDepth.SCHEMA_AND_DATA
    try:
        yield
    finally:
        CONFIG.validation_depth = depth
        reset_config_context()


class BackendFixture:
    name: str

    @property
    def DataFrameSchema(self):
        raise NotImplementedError

    @property
    def Column(self):
        raise NotImplementedError

    @property
    def string_dtype(self):
        raise NotImplementedError

    @property
    def _engine_cls(self):
        raise NotImplementedError

    @property
    def _extra_dtypes(self) -> dict:
        return {}

    def dtype(self, data_type):
        if isinstance(data_type, str) and data_type in self._extra_dtypes:
            return self._extra_dtypes[data_type]
        try:
            return self._engine_cls.dtype(data_type).type
        except (TypeError, ModuleNotFoundError):
            return None

    def make_frame(self, data: dict, dtype=None):
        raise NotImplementedError

    def is_narwhals_incompatible(self, dtype) -> bool:
        from pandera.engines.narwhals_engine import Engine as NarwhalsEngine

        try:
            NarwhalsEngine.dtype(self._engine_cls.dtype(dtype))
            return False
        except TypeError:
            return True


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

    @property
    def string_dtype(self):
        from pandera.engines.polars_engine import String

        return String.type

    @property
    def _engine_cls(self):
        from pandera.engines.polars_engine import Engine

        return Engine

    @property
    def _extra_dtypes(self) -> dict:
        import polars as pl

        from pandera.engines import polars_engine as pe

        return {
            "categorical": pe.Category.type,
            "list_utf8": pe.List(inner=pl.Utf8).type,
        }

    def make_frame(self, data: dict, dtype=None):
        import polars as pl

        if dtype is None:
            return pl.DataFrame(data)
        return pl.LazyFrame(
            data,
            orient="row",
            schema={
                "product": pl.Utf8,
                "code": self._engine_cls.dtype(dtype).type,
            },
        )


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

    @property
    def string_dtype(self):
        from pandera.engines.ibis_engine import String

        return String.type

    @property
    def _engine_cls(self):
        from pandera.engines.ibis_engine import Engine

        return Engine

    def make_frame(self, data: dict, dtype=None):
        import ibis

        if dtype is None:
            return ibis.memtable(data)
        import ibis.expr.datatypes as dt

        schema = ibis.schema(
            {
                "product": dt.String(),
                "code": self._engine_cls.dtype(dtype).type,
            }
        )
        return ibis.memtable(data, schema=schema)


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
    return backend.make_frame({"x": [1, 2, 3], "y": [10, 20, 30]})
