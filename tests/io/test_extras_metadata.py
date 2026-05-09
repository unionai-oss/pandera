"""Regression test for the `io` and `all` optional-dependency declarations.

Issue: https://github.com/unionai-oss/pandera/issues/2127

The ``io`` extra exposes ``pandera.io.pandas_io``, which imports pandas at
module top-level. Before this fix, ``pip install 'pandera[io]'`` resolved
without pulling in pandas/numpy on a fresh environment, so the first
``import pandera.io.pandas_io`` raised ``ModuleNotFoundError: pandas`` —
even though the user followed the documented install path.

The ``all`` extra is documented as bundling every backend; the same gap
applied to it, since pandas was inherited transitively via xarray's numpy
declaration but never declared directly.

This test pins the contract by asserting the declarations directly. If a
future refactor moves the io module away from pandas, the test should be
updated to match the new dependency graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

PYPROJECT = (
    Path(__file__).resolve().parents[2] / "pyproject.toml"
)


def _opt_deps() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["optional-dependencies"]


def _has_dep(deps: list[str], name: str) -> bool:
    """True if any element of *deps* declares the given distribution name."""
    name = name.lower()
    for dep in deps:
        # strip version specifiers / extras / markers
        head = dep.split(";")[0].split("[")[0]
        head = head.split(">=")[0].split("==")[0].split("<")[0].split(">")[0]
        if head.strip().lower() == name:
            return True
    return False


def test_io_extra_declares_pandas_and_numpy() -> None:
    deps = _opt_deps()["io"]
    assert _has_dep(deps, "pandas"), deps
    assert _has_dep(deps, "numpy"), deps
    assert _has_dep(deps, "pyyaml"), deps


def test_all_extra_declares_pandas_and_numpy() -> None:
    deps = _opt_deps()["all"]
    assert _has_dep(deps, "pandas"), deps
    assert _has_dep(deps, "numpy"), deps
