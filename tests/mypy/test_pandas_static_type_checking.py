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

import pandera.pandas as pa
from tests.mypy.pandas_modules import pandas_dataframe

test_module_dir = Path(os.path.dirname(__file__))


def _get_mypy_errors(
    module_name: str,
    stdout,
) -> typing.List[typing.Dict[str, str]]:
    """Parse line number and error message."""
    errors: typing.List[typing.Dict[str, str]] = []
    # last line is summary of errors
    for error in [x for x in stdout.split("\n") if x != ""][:-1]:
        regex = (
            r".+{}:".format(module_name.replace(".", r"\."))
            + r"(?P<lineno>\d+): error: (?P<msg>.+)  \[(?P<errcode>.+)\]"
        )
        matches = re.match(regex, error)
        if matches is not None:
            match_dict = matches.groupdict()
            errors.append(
                {
                    "msg": match_dict["msg"],
                    "errcode": match_dict["errcode"],
                }
            )
    return errors


PANDAS_DATAFRAME_ERRORS = [
    {"msg": "^Argument 1 to .+ has incompatible type", "errcode": "arg-type"},
    {"msg": "^Incompatible return value type", "errcode": "return-value"},
    {"msg": "^Argument 1 to .+ has incompatible type", "errcode": "arg-type"},
    {"msg": "^Argument 1 to .+ has incompatible type", "errcode": "arg-type"},
    {"msg": "^Incompatible return value type", "errcode": "return-value"},
]


@pytest.mark.parametrize(
    ["config_file", "expected_errors"],
    [("no_plugin.ini", PANDAS_DATAFRAME_ERRORS), ("plugin_mypy.ini", [])],
)
def test_mypy_pandas_dataframe(capfd, config_file, expected_errors) -> None:
    """Test that mypy raises expected errors on pandera-decorated functions."""
    # pylint: disable=subprocess-run-check
    pytest.xfail(
        f"pandas_dataframe.py module is unstable when it comes due to maturing "
        "pandas-stubs library"
    )
    cache_dir = str(test_module_dir / ".mypy_cache" / "test-mypy-default")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(test_module_dir / "modules" / "pandas_dataframe.py"),
            "--cache-dir",
            cache_dir,
            "--config-file",
            str(test_module_dir / "config" / config_file),
        ],
        text=True,
    )
    errors = _get_mypy_errors("pandas_dataframe.py", capfd.readouterr().out)
    assert len(expected_errors) == len(errors)
    for expected, error in zip(expected_errors, errors):
        assert error["errcode"] == expected["errcode"]
        assert expected["msg"] == error["msg"] or re.match(
            expected["msg"], error["msg"]
        )


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
        assert e.failure_cases == "age"


PANDERA_INHERITANCE_ERRORS = [
    {"msg": "Incompatible types in assignment", "errcode": "assignment"}
] * 3

PANDERA_TYPES_ERRORS = [
    {"msg": 'Argument 1 to "fn" has incompatible type', "errcode": "arg-type"},
] * 2

PANDAS_TIME_ERRORS = [
    {
        "msg": 'Argument 1 to "Timedelta" has incompatible type "float"',
        "errcode": "arg-type",
    },
]

PYTHON_SLICE_ERRORS = [
    {"msg": "Slice index must be an integer", "errcode": "misc"},
]

PANDAS_INDEX_ERRORS = [
    {"msg": "Incompatible types in assignment", "errcode": "assignment"},
] * 3

PANDAS_SERIES_ERRORS_NO_PLUGIN = [
    {
        "msg": (
            'Argument 1 to "fn" has incompatible type "Series[float]"; '
            'expected "Series[str]"'
        ),
        "errcode": "arg-type",
    }
]

PANDAS_SERIES_ERRORS_PLUGIN = [
    {
        "msg": (
            'Argument "s" to "fn" has incompatible type "Series[float]"; '
            'expected "Series[str]"'
        ),
        "errcode": "arg-type",
    },
    {
        "msg": (
            'Argument 1 to "fn" has incompatible type "Series[float]"; '
            'expected "Series[str]"'
        ),
        "errcode": "arg-type",
    },
]


@pytest.mark.parametrize(
    "module,config,expected_errors",
    [
        [
            "pandera_inheritance.py",
            "no_plugin.ini",
            PANDERA_INHERITANCE_ERRORS,
        ],
        ["pandera_inheritance.py", "plugin_mypy.ini", []],
        ["pandera_types.py", "no_plugin.ini", PANDERA_TYPES_ERRORS],
        ["pandera_types.py", "plugin_mypy.ini", []],
        ["pandas_concat.py", "no_plugin.ini", []],
        ["pandas_concat.py", "plugin_mypy.ini", []],
        ["pandas_time.py", "no_plugin.ini", PANDAS_TIME_ERRORS],
        ["pandas_time.py", "plugin_mypy.ini", PANDAS_TIME_ERRORS],
        ["python_slice.py", "no_plugin.ini", PYTHON_SLICE_ERRORS],
        ["python_slice.py", "plugin_mypy.ini", []],
        ["pandas_index.py", "no_plugin.ini", []],
        ["pandas_index.py", "plugin_mypy.ini", []],
        ["pandas_series.py", "no_plugin.ini", PANDAS_SERIES_ERRORS_NO_PLUGIN],
        ["pandas_series.py", "plugin_mypy.ini", PANDAS_SERIES_ERRORS_PLUGIN],
    ],
)
def test_pandas_stubs_false_positives(
    capfd,
    module,
    config,
    expected_errors,
) -> None:
    """Test pandas-stubs type stub false positives."""
    xfail_modules = {
        "pandera_inheritance.py",
        "pandera_types.py",
        "pandas_time.py",
        "pandas_index.py",
        "pandas_series.py",
    }
    if module in xfail_modules:
        pytest.xfail(
            f"{xfail_modules} are unstable when it comes due to maturing "
            "pandas-stubs library"
        )

    cache_dir = str(
        test_module_dir
        / ".mypy_cache"
        / f'{module.replace(".", "_")}-{config.replace(".", "_")}'
    )

    commands = [
        sys.executable,
        "-m",
        "mypy",
        str(test_module_dir / "pandas_modules" / module),
        "--cache-dir",
        cache_dir,
        "--config-file",
        str(test_module_dir / "config" / config),
    ]
    # pylint: disable=subprocess-run-check
    subprocess.run(commands, text=True)
    resulting_errors = _get_mypy_errors(module, capfd.readouterr().out)
    assert len(expected_errors) == len(resulting_errors)
    for expected, error in zip(expected_errors, resulting_errors):
        assert error["errcode"] == expected["errcode"]
        assert expected["msg"] == error["msg"] or re.match(
            expected["msg"], error["msg"]
        )


@pytest.mark.parametrize(
    "module",
    [
        "pandera_inheritance",
        "pandera_types",
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
    xfail_modules = {
        "pandas_series",
    }
    if module in xfail_modules:
        pytest.xfail(
            f"{xfail_modules} are unstable when it comes due to maturing "
            "pandas-stubs library"
        )
    importlib.import_module(f"tests.mypy.pandas_modules.{module}")
