# /// script
# dependencies = ["nox"]
# ///

"""Nox sessions."""

# isort: skip_file
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

import nox
from nox import Session


nox.options.sessions = (
    "requirements",
    "tests",
    "docs",
)

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
PANDAS_VERSIONS = ["2.1.1", "2.2.3"]
PYDANTIC_VERSIONS = ["1.10.11", "2.10.6"]
PACKAGE = "pandera"
SOURCE_PATHS = PACKAGE, "tests", "noxfile.py"
REQUIREMENT_PATH = "requirements.txt"
EXTRAS_REQUIRING_PANDAS = frozenset(
    [
        "io",
        "mypy",
        "fastapi",
        "hypotheses",
        "strategies",
    ]
)

CI_RUN = os.environ.get("CI") == "true"
if CI_RUN:
    print("Running on CI")
else:
    print("Running locally")

LINE_LENGTH = 79


PYPROJECT = nox.project.load_toml("pyproject.toml")
OPTIONAL_DEPENDENCIES = [*PYPROJECT["project"]["optional-dependencies"]]


def _pyproject_requirements() -> Dict[str, List[str]]:
    """Load requirements from setup.py."""
    return {
        "core": PYPROJECT["project"]["dependencies"],
        **PYPROJECT["project"]["optional-dependencies"],
    }


def _dev_requirements() -> List[str]:
    """Load requirements from file."""
    with open(REQUIREMENT_PATH, encoding="utf-8") as req_file:
        reqs = []
        for req in req_file.readlines():
            if req.startswith("#"):
                continue
            reqs.append(req.strip())
        return reqs


def _generate_pip_deps_from_conda(
    session: Session, compare: bool = False
) -> None:
    args = ["scripts/generate_pip_deps_from_conda.py"]
    if compare:
        args.append("--compare")
    session.run("python", *args)


@nox.session(venv_backend="uv", python=PYTHON_VERSIONS)
def requirements(session: Session) -> None:  # pylint:disable=unused-argument
    """Check that setup.py requirements match requirements.in"""
    session.install("pyyaml")
    try:
        _generate_pip_deps_from_conda(session, compare=True)
    except nox.command.CommandFailed as err:
        _generate_pip_deps_from_conda(session)
        print(f"{REQUIREMENT_PATH} has been re-generated ✨ 🍰 ✨")
        raise err

    ignored_pkgs = {"black", "pandas", "pandas-stubs", "modin"}
    mismatched = []

    # only compare package versions, not python version markers.
    str_dev_reqs = _dev_requirements()
    requirements = _pyproject_requirements()
    for extra, reqs in requirements.items():
        for req in reqs:
            req = req.split(";")[0].strip()
            if req not in ignored_pkgs and req not in str_dev_reqs:
                mismatched.append(f"{extra}: {req}")

    if mismatched:
        print(
            f"Packages {mismatched} defined in pyproject.toml "
            f"do not match {REQUIREMENT_PATH}."
        )
        print(
            "Modify environment.yml, "
            f"then run 'nox -s requirements' to generate {REQUIREMENT_PATH}"
        )
        sys.exit(1)


def _testing_requirements(
    session: Session,
    extra: Optional[str] = None,
    pandas: Optional[str] = None,
    pydantic: Optional[str] = None,
) -> list[str]:

    pandas = pandas or PANDAS_VERSIONS[-1]
    pydantic = pydantic or PYDANTIC_VERSIONS[-1]

    _requirements = PYPROJECT["project"]["dependencies"]
    if extra is not None:
        _requirements += PYPROJECT["project"]["optional-dependencies"][extra]

    # some of the extras are only supported with the pandas extra
    if extra in EXTRAS_REQUIRING_PANDAS:
        _requirements.extend(
            PYPROJECT["project"]["optional-dependencies"]["pandas"]
        )

    _requirements = list(set(_requirements))

    _numpy: Optional[str] = None
    if pandas != "2.2.3" or (
        extra == "pyspark" and session.python in ("3.9", "3.10")
    ):
        # constrain numpy < 2 for older versions of pandas and pyspark on py3.9 and py3.10
        _numpy = "< 2"

    _updated_requirements = []
    for req in _requirements:
        req = req.strip()
        if req == "pandas" or req.startswith("pandas "):
            req = f"pandas=={pandas}"
        if req == "pydantic" or req.startswith("pydantic "):
            req = f"pydantic=={pydantic}"
        if req.startswith("numpy") and _numpy is not None:
            print("adding numpy constraint <2")
            req = f"{req}, {_numpy}"
        if (
            req == "polars"
            or req.startswith("polars ")
            and sys.platform == "darwin"
        ):
            req = "polars-lts-cpu"
        # for some reason uv will try to install an old version of dask,
        # have to specifically pin dask[dataframe] to a higher version
        if (
            req == "dask[dataframe]" or req.startswith("dask[dataframe] ")
        ) and session.python in ("3.9", "3.10", "3.11"):
            req = "dask[dataframe]>=2023.9.2"

        if req not in _updated_requirements:
            _updated_requirements.append(req)

    return [
        *_updated_requirements,
        *nox.project.dependency_groups(PYPROJECT, *["dev", "testing", "docs"]),
    ]


# the base module with no extras
EXTRA_PYTHON_PYDANTIC = [(None, None, None)]
for extra in OPTIONAL_DEPENDENCIES:
    if extra == "pandas":
        # Only test upper and lower bounds of pandas and pydantic with the
        # pandas extra. The other dataframe library intregations assume either
        # no pandas version, latest supported pandas version, and latest
        # pydantic version. None of the other dataframe libraries use the
        # pydantic integration.
        EXTRA_PYTHON_PYDANTIC.extend(
            [
                (extra, pandas, pydantic)
                for pandas in PANDAS_VERSIONS
                for pydantic in PYDANTIC_VERSIONS
            ]
        )
    else:
        EXTRA_PYTHON_PYDANTIC.append(
            (extra, PANDAS_VERSIONS[-1], PYDANTIC_VERSIONS[-1])
        )


@nox.session(venv_backend="uv", python=PYTHON_VERSIONS)
@nox.parametrize("extra, pandas, pydantic", EXTRA_PYTHON_PYDANTIC)
def tests(
    session: Session,
    extra: Optional[str] = None,
    pandas: Optional[str] = None,
    pydantic: Optional[str] = None,
) -> None:
    """Run the test suite."""

    requirements = _testing_requirements(session, extra, pandas, pydantic)
    session.install(*requirements)
    session.install("-e", ".", "--config-settings", "editable_mode=compat")
    session.run("uv", "pip", "list")

    env = {}
    test_dir = "base" if extra is None else extra

    if extra and extra.startswith("modin"):
        modin_split = extra.split("-")
        if len(modin_split) == 1:
            # default to ray
            engine = "ray"
        else:
            extra, engine = modin_split
            test_dir = extra
        if engine not in {"dask", "ray"}:
            raise ValueError(f"{engine} is not a valid modin engine")
        env = {"CI_MODIN_ENGINES": engine}

    if session.posargs:
        args = session.posargs
    else:
        path = f"tests/{test_dir}/" if extra != "all" else "tests"
        args = []
        if extra == "strategies":
            profile = "ci"
            # enable threading via pytest-xdist
            args = [
                "-n=auto",
                "-q",
                f"--hypothesis-profile={profile}",
            ]
        args += [
            f"--cov={PACKAGE}",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-append",
            "--verbosity=10",
        ]
        if not CI_RUN:
            args.append("--cov-report=html")
        args.append(path)

    session.run("pytest", *args, env=env)


@nox.session(venv_backend="uv", python=PYTHON_VERSIONS)
def docs(session: Session) -> None:
    """Build the documentation."""
    # this is needed until ray and geopandas are supported on python 3.10

    session.install("-e", ".")
    session.install(
        *_testing_requirements(session, extra="all"),
        *nox.project.dependency_groups(PYPROJECT, "dev", "testing", "docs"),
    )
    session.chdir("docs")

    # build html docs
    if not CI_RUN and not session.posargs:
        shutil.rmtree("_build", ignore_errors=True)
        shutil.rmtree(
            os.path.join("source", "reference", "generated"),
            ignore_errors=True,
        )
        for builder in ["doctest", "html"]:
            session.run(
                "sphinx-build",
                "-W",
                "-T",
                f"-b={builder}",
                "-d",
                os.path.join("_build", "doctrees", ""),
                "source",
                os.path.join("_build", builder, ""),
            )
    else:
        shutil.rmtree(os.path.join("_build"), ignore_errors=True)
        args = session.posargs or [
            "-v",
            "-W",
            "-E",
            "-b=doctest",
            "source",
            "_build",
        ]
        session.run(
            "sphinx-build",
            *args,
        )

    session.run("xdoctest", PACKAGE, "--quiet")


if __name__ == "__main__":
    nox.main()
