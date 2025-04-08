"""Unit tests for granular control based on validation depth."""

import pytest

from pandera.backends.base import CoreCheckResult
from pandera.config import ValidationDepth, ValidationScope, config_context
from pandera.validation_depth import validate_scope


def custom_backend():
    class CustomBackend:

        # pylint: disable=unused-argument
        @validate_scope(ValidationScope.SCHEMA)
        def check_schema(self, check_obj):
            # core check result is passed as True when validation scope doesn't
            # include schema checks
            return CoreCheckResult(passed=False)

        # pylint: disable=unused-argument
        @validate_scope(ValidationScope.DATA)
        def check_data(self, check_obj):
            # core check result is passed as True when validation scope doesn't
            # include data checks
            return CoreCheckResult(passed=False)

    return CustomBackend()


@pytest.mark.parametrize(
    "validation_depth,expected",
    [
        [ValidationDepth.SCHEMA_ONLY, [False, True]],
        [ValidationDepth.DATA_ONLY, [True, False]],
        [ValidationDepth.SCHEMA_AND_DATA, [False, False]],
        [None, [False, False]],
    ],
)
def test_validate_scope(validation_depth, expected):

    with config_context(validation_depth=validation_depth):
        backend = custom_backend()
        schema_result = backend.check_schema("foo")
        data_result = backend.check_data("foo")
        results = [schema_result.passed, data_result.passed]
        assert results == expected
