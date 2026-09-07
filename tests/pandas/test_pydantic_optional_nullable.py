"""
Regression tests for Optional PydanticModel fields treated as required
(GitHub issue #2406).

When using ``Config.dtype = PydanticModel(Model)`` in a DataFrameModel,
optional fields (e.g. ``rating: Optional[int] = None``) were treated as
non-nullable, causing validation to fail with "non-nullable series
contains null values" even though the field is optional in the pydantic
model.

The fix: ``_build_pydantic_column`` now inspects the pydantic field's
annotation and default to determine whether the column should be nullable.
"""

from typing import Optional

import pandas as pd
import pytest
from pydantic import BaseModel, PositiveInt

import pandera.pandas as pa
from pandera.engines.pandas_engine import PydanticModel


class BookWithOptionalRating(BaseModel):
    title: str
    rating: PositiveInt | None = None


class BookWithRequiredRating(BaseModel):
    title: str
    rating: PositiveInt


class OptionalBookSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(BookWithOptionalRating)


class RequiredBookSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(BookWithRequiredRating)


class TestOptionalPydanticFieldNullable:
    """Optional pydantic fields should produce nullable columns."""

    def test_optional_field_is_nullable_in_schema(self):
        """The schema should mark optional pydantic fields as nullable."""
        schema = OptionalBookSchema.to_schema()
        rating_col = schema.columns.get("rating")
        assert rating_col is not None, "rating column should exist"
        assert rating_col.nullable is True, (
            "Optional pydantic field should be nullable"
        )

    def test_required_field_is_not_nullable_in_schema(self):
        """The schema should mark required pydantic fields as non-nullable."""
        schema = RequiredBookSchema.to_schema()
        rating_col = schema.columns.get("rating")
        assert rating_col is not None, "rating column should exist"
        assert rating_col.nullable is False, (
            "Required pydantic field should not be nullable"
        )

    def test_title_field_is_not_nullable(self):
        """A required str field should not be nullable."""
        schema = OptionalBookSchema.to_schema()
        title_col = schema.columns.get("title")
        assert title_col is not None
        assert title_col.nullable is False

    def test_missing_optional_column_accepted(self):
        """A missing optional column should not raise a validation error."""
        df = pd.DataFrame({"title": ["Dune", "Foundation"]})
        result = OptionalBookSchema.validate(df)
        assert len(result) == 2

    def test_optional_column_with_none_values(self):
        """None values in an optional column should be accepted."""
        df = pd.DataFrame(
            {"title": ["Dune", "Foundation"], "rating": [None, None]}
        )
        result = OptionalBookSchema.validate(df)
        assert len(result) == 2
