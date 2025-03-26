#!/usr/bin/env python
# This script is a modified copy of generate_pip_deps_from_conda.py from
# pandas and is distributed under the terms of the BSD 3 License that can be
# found at: https://github.com/pandas-dev/pandas/blob/main/LICENSE
"""
Convert the conda environment.yml to the pip requirements.in,
or check that they have the same packages (for the CI)

Usage:

    Generate `requirements-dev.in`
    $ ./generate_pip_deps_from_conda

    Compare and fail (exit status != 0) if `requirements-dev.in` has not been
    generated with this script:
    $ ./generate_pip_deps_from_conda --compare
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

EXCLUDE = {"python"}
RENAME: Dict[str, str] = {}

REPO_PATH = Path(__file__).resolve().absolute().parents[1]
CONDA_REQUIREMENTS_FILE = REPO_PATH / "environment.yml"
PIP_REQUIREMENTS_FILE = REPO_PATH / "requirements.txt"


def conda_package_to_pip(package: str) -> Optional[str]:
    """
    Convert a conda package to its pip equivalent.

    In most cases they are the same, those are the exceptions:
    - Packages that should be excluded (in `EXCLUDE`)
    - Packages that should be renamed (in `RENAME`)
    - A package requiring a specific version, in conda is defined with a single
      equal (e.g. ``pandas=1.0``) and in pip with two (e.g. ``pandas==1.0``)
    """
    package = re.sub("(?<=[^<>])=", "==", package).strip()

    for compare in ("<=", ">=", "=="):
        if compare not in package:
            continue
        pkg, version = package.split(compare)
        pkg = pkg.strip()

        if pkg in EXCLUDE:
            return None

        if pkg in RENAME:
            return "".join((RENAME[pkg], compare, version))

        break

    if package in EXCLUDE:
        return None

    if package in RENAME:
        return RENAME[package]

    return package


def main(conda_file: Path, pip_file: Path, compare: bool = False) -> bool:
    """
    Generate the pip dependencies file from the conda file, or compare that
    they are synchronized (``compare=True``).

    Parameters
    ----------
    conda_file : Path
        Path to the conda file with dependencies (e.g. `environment.yml`).
    pip_file : Path
        Path to the pip file with dependencies (e.g. `requirements-dev.txt`).
    compare : bool, default False
        Whether to generate the pip file (``False``) or to compare if the
        pip file has been generated with this script and the last version
        of the conda file (``True``).

    Returns
    -------
    bool
        True if the comparison fails, False otherwise
    """
    with open(conda_file) as conda_fd:
        deps = yaml.safe_load(conda_fd)["dependencies"]

    pip_deps: List[str] = []
    for dep in deps:
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            if conda_dep is not None:
                pip_deps.append(conda_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            pip_deps += dep["pip"]
        else:
            raise ValueError(f"Unexpected dependency {dep}")

    fname = conda_file.name
    header = (
        f"# This file is auto-generated from {fname}, do not modify.\n"
        "# See that file for comments about the need/usage of "
        "each dependency.\n\n"
    )
    pip_content = header + "\n".join(pip_deps) + "\n"

    if compare:
        return pip_file.read_text().strip() != pip_content.strip()
    pip_file.write_text(pip_content)
    return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="convert (or compare) conda file to pip"
    )
    argparser.add_argument(
        "--compare",
        action="store_true",
        help="compare whether the two files are equivalent",
    )
    argparser.add_argument(
        "--azure",
        action="store_true",
        help="show the output in azure-pipelines format",
    )
    args = argparser.parse_args()

    res = main(
        CONDA_REQUIREMENTS_FILE, PIP_REQUIREMENTS_FILE, compare=args.compare
    )
    if res:
        msg = (
            f"`{PIP_REQUIREMENTS_FILE}` has to be generated with `{sys.argv[0]}` after "
            f"`{CONDA_REQUIREMENTS_FILE}` is modified.\n"
        )
        if args.azure:
            msg = f"##vso[task.logissue type=error;sourcepath=requirements-in.txt]{msg}"
        sys.stderr.write(msg)
    sys.exit(res)
