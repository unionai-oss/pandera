# /// script
# dependencies = ["nox"]
# ///

"""Nox sessions."""

# isort: skip_file
import os
import re
import shutil
import sys
import tempfile
from typing import Dict, List

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

import nox
from nox import Session


nox.options.sessions = (
    "tests",
    "docs",
    "doctests",
)

DEFAULT_PYTHON = "3.9"
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
PANDAS_VERSIONS = ["2.1.1", "2.2.3"]
PYDANTIC_VERSIONS = ["1.10.11", "2.10.6"]

PACKAGE = "pandera"

SOURCE_PATHS = PACKAGE, "tests", "noxfile.py"
REQUIREMENT_PATH = "requirements.in"
ALWAYS_USE_PIP = {
    "furo",
    "ray",
    "types-click",
    "types-pyyaml",
    "types-setuptools",
}

CI_RUN = os.environ.get("CI") == "true"
if CI_RUN:
    print("Running on CI")
else:
    print("Running locally")

LINE_LENGTH = 79


def _build_setup_requirements() -> Dict[str, List[str]]:
    """Load requirments from setup.py."""
    # read pyproject.toml to get optional dependencies

    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    # Get core dependencies
    core_deps = pyproject["project"]["dependencies"]

    # Get optional dependencies
    optional_deps = pyproject["project"]["optional-dependencies"]
    return {
        "core": [req for req in core_deps],
        **{
            extra: [req for req in deps]
            for extra, deps in optional_deps.items()
        },
    }


def _build_dev_requirements() -> List[str]:
    """Load requirements from file."""
    with open(REQUIREMENT_PATH, encoding="utf-8") as req_file:
        reqs = []
        for req in req_file.readlines():
            reqs.append(req.strip())
        return reqs


SETUP_REQUIREMENTS: Dict[str, List[str]] = _build_setup_requirements()
DEV_REQUIREMENTS: List[str] = _build_dev_requirements()


def _build_requires() -> Dict[str, Dict[str, str]]:
    """Return a dictionary of requirements {EXTRA_NAME: {PKG_NAME:PIP_SPECS}}.

    Adds fake extras "core" and "all".
    """
    extras = {
        extra: reqs
        for extra, reqs in SETUP_REQUIREMENTS.items()
        if extra not in ("core")
    }
    requires = {
        "core": SETUP_REQUIREMENTS["core"],
        "all": [
            *SETUP_REQUIREMENTS["core"],
            *extras["all"],
            *extras["dev"],
            *extras["docs"],
        ],
    }
    requires.update(  # add extras
        {
            extra_name: [*pkgs, *requires["core"]]
            for extra_name, pkgs in extras.items()
            if extra_name != "all"
        }
    )
    return requires


REQUIRES: Dict[str, Dict[str, str]] = _build_requires()

CONDA_ARGS = [
    "--channel=conda-forge",
    "--update-specs",
]


def extract_requirement_name(spec: str) -> str:
    """
    Extract name of requirement from dependency string.
    """
    # Assume name is everything up to the first invalid character
    match = re.match(r"^[A-Za-z0-9-_]*", spec.strip())
    if not match:
        raise ValueError(f"Cannot parse requirement {spec!r}")
    return match[0]


def conda_install(session: Session, *args):
    """Use mamba to install conda dependencies."""
    run_args = [
        "install",
        "--yes",
        *CONDA_ARGS,
        "--prefix",
        session.virtualenv.location,  # type: ignore
        *args,
    ]

    # By default, all dependencies are re-installed from scratch with each
    # session. Specifying external=True allows access to cached packages, which
    # decreases runtime of the test sessions.
    try:
        session.run(
            *["mamba", *run_args],
            external=True,
        )
    # pylint: disable=broad-except
    except Exception:
        session.run(
            *["conda", *run_args],
            external=True,
        )


# def install_extras(
#     session: Session,
#     extra: str = "core",
#     force_pip: bool = False,
#     pandas: str = "latest",
#     pandas_stubs: bool = True,
# ) -> None:
#     """Install dependencies."""

#     if isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
#         # skip this step if there's no virtual environment specified
#         session.run("pip", "install", "-e", ".", "--no-deps")
#         return

#     specs, pip_specs = [], []
#     pandas_version = "" if pandas == "latest" else f"=={pandas}"
#     for spec in REQUIRES[extra].values():
#         req_name = extract_requirement_name(spec)
#         if req_name == "pandas-stubs" and not pandas_stubs:
#             # this is a temporary measure until all pandas-related mypy errors
#             # are addressed
#             continue

#         req = Requirement(spec)  # type: ignore

#         # this is needed until ray is supported on python 3.10
#         # pylint: disable=line-too-long
#         if req.name in {"ray", "geopandas"} and session.python == "3.10":  # type: ignore[attr-defined]  # noqa
#             continue

#         if req.name in ALWAYS_USE_PIP:  # type: ignore[attr-defined]
#             pip_specs.append(spec)
#         elif req_name == "pandas" and pandas != "latest":
#             specs.append(f"pandas~={pandas}")
#         else:
#             specs.append(
#                 spec if spec != "pandas" else f"pandas{pandas_version}"
#             )
#     if extra in {"core", "pyspark", "modin", "fastapi"}:
#         specs.append(REQUIRES["all"]["hypothesis"])

#     # CI installs conda dependencies, so only run this for local runs
#     if (
#         isinstance(session.virtualenv, nox.virtualenv.CondaEnv)
#         and not force_pip
#         and not CI_RUN
#     ):
#         print("using conda installer")
#         conda_install(session, *specs)
#     else:
#         print("using pip installer")
#         session.install(*specs)

#     # always use pip for these packages)
#     session.install(*pip_specs)
#     session.install("-e", ".", "--no-deps")  # install pandera


def _generate_pip_deps_from_conda(
    session: Session, compare: bool = False
) -> None:
    args = ["scripts/generate_pip_deps_from_conda.py"]
    if compare:
        args.append("--compare")
    session.run("python", *args)


@nox.session(python=PYTHON_VERSIONS)
def requirements(session: Session) -> None:  # pylint:disable=unused-argument
    """Check that setup.py requirements match requirements-dev.txt"""
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
    str_dev_reqs = [str(x) for x in DEV_REQUIREMENTS]
    for extra, reqs in SETUP_REQUIREMENTS.items():
        for req in reqs:
            req = req.split(";")[0].strip()
            if req not in ignored_pkgs and req not in str_dev_reqs:
                mismatched.append(f"{extra}: {req}")

    if mismatched:
        print(
            f"Packages {mismatched} defined in setup.py "
            + f"do not match {REQUIREMENT_PATH}."
        )
        print(
            "Modify environment.yml, "
            + f"then run 'nox -s requirements' to generate {REQUIREMENT_PATH}"
        )
        sys.exit(1)


PYTHON_PANDAS_PARAMETER = [
    (python, pandas)
    for python in PYTHON_VERSIONS
    for pandas in PANDAS_VERSIONS
]


def _get_pinned_requirements(
    session: Session, pandas: str, pydantic: str, extra: str
) -> None:
    _requirements = REQUIRES["all"]
    _pinned_requirements = []

    _numpy: str | None = None
    if pandas != "2.2.2":
        _numpy = "< 2"

    for req in _requirements:
        req = req.strip()
        if req == "pandas" or req.startswith("pandas "):
            req = f"pandas=={pandas}\n"
        if req == "pydantic" or req.startswith("pydantic "):
            req = f"pydantic=={pydantic}\n"
        if req.startswith("numpy") and _numpy is not None:
            print("adding numpy constraint <2")
            req = f"{req}, {_numpy}\n"
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
            req = "dask[dataframe]>=2023.9.2\n"

        if req not in _pinned_requirements:
            _pinned_requirements.append(req)

    return _pinned_requirements


EXTRA_NAMES = [
    extra
    for extra in REQUIRES
    if (
        extra != "all"
        and "python_version" not in extra
        and extra not in {"modin", "dev", "docs"}
    )
]


def _install_extras(
    session: Session, extra: str, pandas: str, pydantic: str
) -> None:
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("uv")
        session.run(
            "uv",
            "pip",
            "install",
            "--upgrade",
            *_get_pinned_requirements(session, pandas, pydantic, extra),
        )


@nox.session
@nox.parametrize("python,pandas", PYTHON_PANDAS_PARAMETER)
@nox.parametrize("pydantic", PYDANTIC_VERSIONS)
@nox.parametrize("extra", EXTRA_NAMES)
def tests(session: Session, pandas: str, pydantic: str, extra: str) -> None:
    """Run the test suite."""

    _install_extras(session, extra, pandas, pydantic)
    session.run("uv", "pip", "list")

    env = {}
    if extra.startswith("modin"):
        extra, engine = extra.split("-")
        if engine not in {"dask", "ray"}:
            raise ValueError(f"{engine} is not a valid modin engine")
        env = {"CI_MODIN_ENGINES": engine}

    if session.posargs:
        args = session.posargs
    else:
        path = f"tests/{extra}/" if extra != "all" else "tests"
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


@nox.session(python=PYTHON_VERSIONS)
def doctests(session: Session) -> None:
    """Build the documentation."""
    _install_extras(session, "all", PANDAS_VERSIONS[-1], PYDANTIC_VERSIONS[-1])
    if session.python == "3.12":
        # skip 3.12 because of pyspark depends on distutils and 3.12 dropped it
        session.skip()
    session.run("xdoctest", PACKAGE, "--quiet")


@nox.session(python=PYTHON_VERSIONS)
def docs(session: Session) -> None:
    """Build the documentation."""
    # this is needed until ray and geopandas are supported on python 3.10

    _install_extras(session, "all", PANDAS_VERSIONS[-1], PYDANTIC_VERSIONS[-1])
    session.run("uv", "sync", "--active")
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
        session.run("sphinx-build", *args)


if __name__ == "__main__":
    nox.main()
