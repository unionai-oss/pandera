"""Test that pandera schemas are thread safe."""

import pandas as pd
import numpy as np
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
