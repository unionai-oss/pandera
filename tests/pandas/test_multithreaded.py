"""Test that pandera schemas are thread safe."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import pandera.pandas as pa


class Model(pa.DataFrameModel):
    time: pa.typing.Series[np.float32] = pa.Field(coerce=True)


def validate_df(df):
    validated_df = Model.to_schema().validate(df)
    assert validated_df.dtypes["time"] == np.float32
    return validated_df


def test_single_thread():
    df = pd.DataFrame({"time": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
    validate_df(df)


def test_multithreading():
    df = pd.DataFrame({"time": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
    n_tries = 10
    total = 8
    n_jobs = 4

    for _ in range(n_tries):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(validate_df)(df) for _ in range(total)
        )
        for res in results:
            assert res.dtypes["time"] == np.float32


def test_concurrent_cold_schema_validate_does_not_skip_coerce(monkeypatch):
    schema = pa.DataFrameSchema({"x": pa.Column(float, coerce=True)})
    df = pd.DataFrame({"x": [1, 2, 3]})

    component_check_started = threading.Event()
    release_component_check = threading.Event()
    blocked_once = False
    block_lock = threading.Lock()
    original_validate = pa.Column.validate

    def blocking_validate(self, *args, **kwargs):
        nonlocal blocked_once
        with block_lock:
            should_block = not blocked_once and self.name == "x"
            if should_block:
                blocked_once = True

        if should_block:
            component_check_started.set()
            assert release_component_check.wait(timeout=5)

        return original_validate(self, *args, **kwargs)

    monkeypatch.setattr(pa.Column, "validate", blocking_validate)

    def validate():
        return schema.validate(df.copy())

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_validation = executor.submit(validate)
        try:
            assert component_check_started.wait(timeout=5)
            second_result = executor.submit(validate).result(timeout=5)
        finally:
            release_component_check.set()

        first_result = first_validation.result(timeout=5)

    assert first_result.dtypes["x"] == np.float64
    assert second_result.dtypes["x"] == np.float64
    assert schema.columns["x"].coerce
