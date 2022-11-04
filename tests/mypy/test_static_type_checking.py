# pylint: skip-file
"""Unit tests for static type checking of dataframes.

This module uses subprocess and the pytest.capdf fixture to capture the output
of calling mypy on the python modules in the tests/core/static folder.
"""

import importlib
import os
import re
import subprocess
import sys
import typing
from pathlib import Path

import pytest

import pandera as pa
from tests.mypy.modules import pandas_dataframe

test_module_dir = Path(os.path.dirname(__file__))


def _get_mypy_errors(stdout) -> typing.List[typing.Dict[str, str]]:
    """Parse line number and error message."""
    errors: typing.List[typing.Dict[str, str]] = []
    # last line is summary of errors
    for error in [x for x in stdout.split("\n") if x != ""][:-1]:
        matches = re.match(
            r".+\.py:(?P<lineno>\d+): error: (?P<msg>.+)  \[(?P<errcode>.+)\]",
            error,
        )
        if matches is not None:
            match_dict = matches.groupdict()
            errors.append(
                {
                    "msg": match_dict["msg"],
                    "errcode": match_dict["errcode"],
                }
            )
    return errors


def test_mypy_pandas_dataframe(capfd) -> None:
    """Test that mypy raises expected errors on pandera-decorated functions."""
    # pylint: disable=subprocess-run-check
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(test_module_dir / "modules" / "pandas_dataframe.py"),
        ],
        text=True,
    )
    errors = _get_mypy_errors(capfd.readouterr().out)
    # assert error messages on particular lines of code
    assert errors[0]["errcode"] == "arg-type"
    assert re.match(
        "^Argument 1 to .+ has incompatible type", errors[0]["msg"]
    )

    assert errors[1]["errcode"] == "return-value"
    assert re.match("^Incompatible return value type", errors[1]["msg"])

    assert errors[2]["errcode"] == "arg-type"
    assert re.match(
        '^Argument 1 to ".+" has incompatible type', errors[2]["msg"]
    )

    assert errors[3]["errcode"] == "arg-type"
    assert re.match(
        '^Argument 1 to ".+" has incompatible type', errors[3]["msg"]
    )

    assert errors[4]["errcode"] == "return-value"
    assert re.match("^Incompatible return value type", errors[4]["msg"])


@pytest.mark.parametrize(
    "fn",
    [
        pandas_dataframe.fn_mutate_inplace,
        pandas_dataframe.fn_assign_and_get_index,
        pandas_dataframe.fn_cast_dataframe_invalid,
    ],
)
def test_pandera_runtime_errors(fn) -> None:
    """Test that pandera catches cases that mypy doesn't catch."""

    # both functions don't add a required column "age"
    try:
        fn(pandas_dataframe.schema_df)
    except pa.errors.SchemaError as e:
        assert e.failure_cases["failure_case"].item() == "age"


PYTHON_SLICE_ERRORS = [
    {"msg": "Slice index must be an integer or None", "errcode": "misc"},
]

PANDAS_INDEX_ERRORS = [
    {"msg": "Incompatible types in assignment", "errcode": "assignment"},
] * 3

PANDERA_TYPES_ERRORS = [
    {"msg": 'Argument 1 to "fn" has incompatible type', "errcode": "arg-type"},
] * 2

PANDAS_SERIES_ERRORS = [
    {
        "msg": (
            'Argument 1 to "fn" has incompatible type "Series[float]"; '
            'expected "Series[str]"'
        ),
        "errcode": "arg-type",
    }
]


@pytest.mark.parametrize(
    "module,config,expected_errors",
    [
        ["pandas_concat.py", None, []],
        ["pandas_concat.py", "plugin_mypy.ini", []],
        ["pandas_time.py", None, []],
        ["pandas_time.py", "plugin_mypy.ini", []],
        ["python_slice.py", None, PYTHON_SLICE_ERRORS],
        ["python_slice.py", "plugin_mypy.ini", PYTHON_SLICE_ERRORS],
        ["pandas_index.py", None, PANDAS_INDEX_ERRORS],
        ["pandas_index.py", "plugin_mypy.ini", PANDAS_INDEX_ERRORS],
        ["pandera_types.py", None, PANDERA_TYPES_ERRORS],
        ["pandera_types.py", "plugin_mypy.ini", PANDERA_TYPES_ERRORS],
        ["pandas_series.py", None, PANDAS_SERIES_ERRORS],
    ],
)
def test_pandas_stubs_false_positives(
    capfd,
    module,
    config,
    expected_errors,
) -> None:
    """Test pandas-stubs type stub false positives."""
    if config is None:
        cache_dir = str(test_module_dir / ".mypy_cache" / "test-mypy-default")
    else:
        cache_dir = str(test_module_dir / ".mypy_cache" / f"test-{config}")

    commands = [
        sys.executable,
        "-m",
        "mypy",
        str(test_module_dir / "modules" / module),
        "--cache-dir",
        cache_dir,
    ]

    if config:
        commands += ["--config-file", str(test_module_dir / "config" / config)]
    # pylint: disable=subprocess-run-check
    subprocess.run(commands, text=True)
    resulting_errors = _get_mypy_errors(capfd.readouterr().out)
    assert len(expected_errors) == len(resulting_errors)
    for expected, error in zip(expected_errors, resulting_errors):
        assert error["errcode"] == expected["errcode"]
        assert expected["msg"] == error["msg"] or re.match(
            expected["msg"], error["msg"]
        )


@pytest.mark.parametrize(
    "module",
    [
        "pandas_concat",
        "pandas_time",
        "python_slice",
        "pandas_index",
        "pandera_types",
        "pandas_series",
    ],
)
def test_pandas_modules_importable(module):
    """Make sure that static type linting modules can be executed."""
    importlib.import_module(f"tests.mypy.modules.{module}")
