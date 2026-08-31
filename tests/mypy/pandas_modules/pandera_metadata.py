# pylint: skip-file
from typing import Any

import pandera.pandas as pa


class Foo(pa.DataFrameModel):
    x: int


model_metadata: dict[Any, Any] = Foo.get_metadata()
model_value = model_metadata[str(Foo)]

schema = Foo.to_schema()
schema_metadata: dict[Any, Any] = schema.get_metadata()
