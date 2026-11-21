"""Benchmark pydantic row-wise validation against native pandera validation.

Compares equivalent schemas over the same dataframe:

* ``PydanticModel``: a pydantic model applied row by row via
  ``pandera.engines.pandas_engine.PydanticModel``.
* native: ``pandera.pandas.DataFrameModel`` with column dtypes and checks.

Usage::

    python scripts/benchmark_pydantic_vs_pandera.py
    python scripts/benchmark_pydantic_vs_pandera.py --sizes 100 1000 10000 --repeat 5
"""

import argparse
import logging
import statistics
import timeit

import numpy as np
import pandas as pd
import pydantic

import pandera.pandas as pa
from pandera.engines.pandas_engine import PydanticModel

# The PydanticModel dtype logs a slow-path warning on every validation call.
logging.getLogger("pandera").setLevel(logging.ERROR)


class Record(pydantic.BaseModel):
    """Pydantic schema."""

    id: int
    price: float = pydantic.Field(ge=0, le=1000)
    quantity: int = pydantic.Field(gt=0)
    name: str


class NativeSchema(pa.DataFrameModel):
    """Pandera schema equivalent to ``Record``."""

    id: int
    price: float = pa.Field(ge=0, le=1000)
    quantity: int = pa.Field(gt=0)
    name: str


# PydanticModel is a dataframe-level dtype, so it needs a DataFrameSchema.
pydantic_schema = pa.DataFrameSchema(dtype=PydanticModel(Record))
native_schema = NativeSchema.to_schema()


def make_df(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Build a valid dataframe with ``n_rows`` rows."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "price": rng.uniform(0, 1000, n_rows),
            "quantity": rng.integers(1, 100, n_rows),
            "name": [f"item-{i}" for i in range(n_rows)],
        }
    )


def time_validate(schema: pa.DataFrameSchema, df: pd.DataFrame, repeat: int):
    """Return the median seconds for one ``schema.validate(df)`` call."""
    runs = timeit.repeat(lambda: schema.validate(df), number=1, repeat=repeat)
    return statistics.median(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[100, 1_000, 10_000, 100_000]
    )
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    # Both schemas must accept the data before timing them.
    check_df = make_df(10)
    pydantic_schema.validate(check_df)
    native_schema.validate(check_df)

    print(
        f"{'rows':>10} {'pydantic (s)':>14} {'native (s)':>12} {'slowdown':>10}"
    )
    for n_rows in args.sizes:
        df = make_df(n_rows)
        t_pyd = time_validate(pydantic_schema, df, args.repeat)
        t_nat = time_validate(native_schema, df, args.repeat)
        print(
            f"{n_rows:>10} {t_pyd:>14.4f} {t_nat:>12.4f} {t_pyd / t_nat:>9.1f}x"
        )


if __name__ == "__main__":
    main()
