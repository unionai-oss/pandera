# Airspeed Velocity Benchmarks for pandera polars PydanticModel coercion
import os

# PydanticModel emits a per-call performance warning. pandera reads this env
# var into its config at import time, so it must be set before importing
# pandera below or the warning will not be silenced.
os.environ["SILENCE_WARNING_PYDANTIC_MODEL"] = "true"

import polars as pl  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import pandera.polars as pa  # noqa: E402
from pandera.engines.polars_engine import PydanticModel  # noqa: E402

N = 50_000


class Record(BaseModel):
    name: str | None = None
    code: str | None = None  # nullable id, None for a long leading run
    xcoord: int | None = None


class RecordSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(Record)
        coerce = True
        strict = False


class PydanticModelCoerce:
    """Benchmark PydanticModel coercion on the success and failure paths.

    Tracks time and peak RSS for both paths so future work on the row-wise
    ``model_dump()`` ``list[dict]`` (the dominant allocation) has a baseline.
    """

    def setup(self):
        # The "code" column is None well past the default infer_schema_length
        # window, then carries a real value, mirroring the bug's data shape.
        valid = [{"name": "A", "code": None, "xcoord": i} for i in range(N)]
        valid.append({"name": "B", "code": "00435L108", "xcoord": N})
        schema = {"name": pl.Utf8, "code": pl.Utf8, "xcoord": pl.Object}

        self.success_df = pl.DataFrame(valid, schema=schema)
        # One invalid row near the end triggers the failure path.
        invalid = valid + [{"name": "C", "code": None, "xcoord": "not-an-int"}]
        self.failure_df = pl.DataFrame(invalid, schema=schema)

    def time_success(self):
        RecordSchema.validate(self.success_df)

    def peakmem_success(self):
        RecordSchema.validate(self.success_df)

    def time_failure(self):
        try:
            RecordSchema.validate(self.failure_df)
        except pa.errors.SchemaError:
            pass

    def peakmem_failure(self):
        try:
            RecordSchema.validate(self.failure_df)
        except pa.errors.SchemaError:
            pass
