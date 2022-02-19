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


def _get_mypy_errors(stdout) -> typing.Dict[int, typing.Dict[str, str]]:
    """Parse line number and error message."""
    errors: typing.Dict[int, typing.Dict[str, typing.Any]] = {}
    # last line is summary of errors
    for error in [x for x in stdout.split("\n") if x != ""][:-1]:
        matches = re.match(
            r".+\.py:(?P<lineno>\d+): error: (?P<msg>.+)  \[(?P<errcode>.+)\]",
            error,
        )
        if matches is not None:
            match_dict = matches.groupdict()
            errors[int(match_dict["lineno"])] = {
                "msg": match_dict["msg"],
                "errcode": match_dict["errcode"],
            }
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
    assert errors[35] == {
        "msg": (
            'Argument 1 to "pipe" of "NDFrame" has incompatible type '
            '"Type[DataFrame[Any]]"; expected '
            '"Union[Callable[..., DataFrame[SchemaOut]], '
            'Tuple[Callable[..., DataFrame[SchemaOut]], str]]"'
        ),
        "errcode": "arg-type",
    }
    assert errors[41] == {
        "msg": (
            "Incompatible return value type (got "
            '"pandas.core.frame.DataFrame", expected '
            '"pandera.typing.pandas.DataFrame[SchemaOut]")'
        ),
        "errcode": "return-value",
    }
    assert errors[54] == {
        "msg": (
            'Argument 1 to "fn" has incompatible type '
            '"pandas.core.frame.DataFrame"; expected '
            '"pandera.typing.pandas.DataFrame[Schema]"'
        ),
        "errcode": "arg-type",
    }
    assert errors[58] == {
        "msg": (
            'Argument 1 to "fn" has incompatible type '
            '"DataFrame[AnotherSchema]"; expected "DataFrame[Schema]"'
        ),
        "errcode": "arg-type",
    }


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


# pylint: disable=line-too-long
PANDAS_CONCAT_FALSE_POSITIVES = {
    13: {
        "msg": 'No overload variant of "concat" matches argument type "Generator[DataFrame, None, None]"',  # noqa
        "errcode": "call-overload",
    },
    16: {
        "msg": 'No overload variant of "concat" matches argument type "Generator[Series, None, None]"',  # noqa
        "errcode": "call-overload",
    },
}

PANDAS_TIME_FALSE_POSITIVES = {
    4: {
        "msg": (
            'Unsupported operand types for + ("Timestamp" and "YearEnd")',
            'No overload variant of "__add__" of "Timestamp" matches argument type "YearEnd"',  # noqa
        ),
        "errcode": "operator",
    },
    6: {
        "msg": 'Missing positional argument "value" in call to "Timedelta"',  # noqa
        "errcode": "call-arg",
    },
    9: {
        "msg": 'Missing positional argument "value" in call to "Timedelta"',  # noqa
        "errcode": "call-arg",
    },
    10: {
        "msg": 'Argument 1 to "Timedelta" has incompatible type "float"; expected "Union[Timedelta, timedelta, timedelta64, str, int]"',  # noqa
        "errcode": "arg-type",
    },
}


@pytest.mark.parametrize(
    "module,config,expected_errors",
    [
        ["pandas_concat.py", None, PANDAS_CONCAT_FALSE_POSITIVES],
        ["pandas_concat.py", "plugin_mypy.ini", {}],
        ["pandas_time.py", None, PANDAS_TIME_FALSE_POSITIVES],
        ["pandas_time.py", "plugin_mypy.ini", {}],
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
    subprocess.run(
        commands,
        text=True,
    )
    resulting_errors = _get_mypy_errors(capfd.readouterr().out)
    for lineno, error in resulting_errors.items():
        assert error["errcode"] == expected_errors[lineno]["errcode"]
        if isinstance(expected_errors[lineno]["msg"], tuple):
            assert error["msg"] in expected_errors[lineno]["msg"]
        else:
            assert error["msg"] == expected_errors[lineno]["msg"]


@pytest.mark.parametrize("module", ["pandas_concat", "pandas_time"])
def test_pandas_modules_importable(module):
    """Make sure that static type linting modules can be executed."""
    importlib.import_module(f"tests.mypy.modules.{module}")
