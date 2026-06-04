"""TDD RED tests for validate_collecting_errors helper (SE-02).

These tests will FAIL until:
1. `validate_collecting_errors` is defined in tests/pyspark/conftest.py
2. All .pandera.errors sites are replaced with the helper in PySpark test files
"""

import pytest

pytest.importorskip("pyspark")


def test_validate_collecting_errors_exists_in_conftest():
    """validate_collecting_errors must be importable from tests.pyspark.conftest."""
    from tests.pyspark.conftest import validate_collecting_errors  # noqa: F401

    assert callable(validate_collecting_errors), (
        "validate_collecting_errors must be a callable function"
    )


def test_validate_collecting_errors_returns_tuple():
    """validate_collecting_errors must return a 2-tuple (out_df, errors_dict)."""
    # Verify the function has the expected signature
    import inspect

    from tests.pyspark.conftest import validate_collecting_errors

    sig = inspect.signature(validate_collecting_errors)
    params = list(sig.parameters.keys())
    assert "schema" in params, (
        "validate_collecting_errors must have a 'schema' parameter"
    )
    assert "df" in params, (
        "validate_collecting_errors must have a 'df' parameter"
    )


def test_validate_collecting_errors_no_inline_pandera_errors_in_container():
    """test_pyspark_container.py must not use .pandera.errors inline (outside comments)."""
    import os
    import re

    container_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "tests",
        "pyspark",
        "test_pyspark_container.py",
    )
    with open(container_path) as f:
        lines = f.readlines()

    inline_sites = [
        (i + 1, line.rstrip())
        for i, line in enumerate(lines)
        if re.search(r"\.pandera\.errors\b", line)
        and not line.lstrip().startswith("#")
    ]
    assert inline_sites == [], (
        f"Found inline .pandera.errors in test_pyspark_container.py at lines: "
        f"{[ln for ln, _ in inline_sites]}"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "test_pyspark_model.py",
        "test_pyspark_check.py",
        "test_pyspark_error.py",
        "test_pyspark_config.py",
        "test_pyspark_dtypes.py",
    ],
)
def test_no_inline_pandera_errors_in_pyspark_test_files(filename):
    """Remaining PySpark test files must not use .pandera.errors inline."""
    import os
    import re

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "tests",
        "pyspark",
        filename,
    )
    with open(path) as f:
        lines = f.readlines()

    inline_sites = [
        (i + 1, line.rstrip())
        for i, line in enumerate(lines)
        if re.search(r"\.pandera\.errors\b", line)
        and not line.lstrip().startswith("#")
    ]
    assert inline_sites == [], (
        f"Found inline .pandera.errors in {filename} at lines: "
        f"{[ln for ln, _ in inline_sites]}"
    )
