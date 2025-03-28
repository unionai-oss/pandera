"""Base schema unit tests."""

import pytest

from pandera.api.base.schema import BaseSchema
from pandera.backends.base import BaseSchemaBackend


class MockSchema(BaseSchema):
    """Mock schema"""


class MockSchemaBackend(BaseSchemaBackend):
    """Mock schema backend"""


def test_get_backend_error():
    """Raise value error when no arguments are passed."""

    schema = MockSchema()
    with pytest.raises(ValueError):
        schema.get_backend()
