"""Nox sessions."""
# isort: skip_file
import os
import shutil
import sys
from typing import Dict, List

# setuptools must be imported before distutils !
import setuptools  # pylint:disable=unused-import  # noqa: F401
from distutils.core import run_setup  # pylint:disable=wrong-import-order

import nox
from nox import Session
from pkg_resources import Requirement, parse_requirements


nox.options.sessions = (
    "requirements",
    "black",
    "isort",
    "lint",
    "mypy",
    "tests",
    "docs",
    "doctests",
)

DEFAULT_PYTHON = "3.8"
PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
PANDAS_VERSIONS = ["1.1.5", "1.3.5", "latest"]

PACKAGE = "pandera"

SOURCE_PATHS = PACKAGE, "tests", "noxfile.py"
REQUIREMENT_PATH = "requirements-dev.txt"
ALWAYS_USE_PIP = {
    "ray",
    "furo",
    "types-click",
    "types-pyyaml",
    "types-pkg_resources",
}

CI_RUN = os.environ.get("CI") == "true"
if CI_RUN:
    print("Running on CI")
else:
    print("Running locally")

LINE_LENGTH = 79


def _build_setup_requirements() -> Dict[str, List[Requirement]]:
    """Load requirments from setup.py."""
    dist = run_setup("setup.py")
    reqs = {"core": dist.install_requires}  # type: ignore
    reqs.update(dist.extras_require)  # type: ignore
    return {
        extra: list(parse_requirements(reqs)) for extra, reqs in reqs.items()
    }


def _build_dev_requirements() -> List[Requirement]:
    """Load requirements from file."""
    with open(REQUIREMENT_PATH, "rt", encoding="utf-8") as req_file:
        reqs = []
        for req in parse_requirements(req_file.read()):
            req.marker = None
            reqs.append(req)
        return reqs


SETUP_REQUIREMENTS: Dict[str, List[Requirement]] = _build_setup_requirements()
DEV_REQUIREMENTS: List[Requirement] = _build_dev_requirements()


def _requirement_to_dict(reqs: List[Requirement]) -> Dict[str, str]:
    """Return a dict {PKG_NAME:PIP_SPECS}."""
    req_dict = {}
    for req in reqs:
        specs = req.specs[0] if req.specs else []
        specs_str = " ".join([req.unsafe_name, *specs]).replace(" ", "")
        req_dict[req.unsafe_name] = specs_str
    return req_dict


def _build_requires() -> Dict[str, Dict[str, str]]:
    """Return a dictionary of requirements {EXTRA_NAME: {PKG_NAME:PIP_SPECS}}.

    Adds fake extras "core" and "all".
    """
    extras = {
        extra: reqs
        for extra, reqs in SETUP_REQUIREMENTS.items()
        if extra not in ("core", "all")
    }
    extras["all"] = DEV_REQUIREMENTS

    optionals = [
        req.project_name
        for extra, reqs in extras.items()
        for req in reqs
        if extra != "all"
    ]
    requires = {"all": _requirement_to_dict(extras["all"])}
    requires["core"] = {
        pkg: specs
        for pkg, specs in requires["all"].items()
        if pkg not in optionals
    }
    requires.update(  # add extras
        {
            extra_name: {**_requirement_to_dict(pkgs), **requires["core"]}
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


def install(session: Session, *args: str):
    """Install dependencies in the appropriate virtual environment
    (conda or virtualenv) and return the type of the environmment."""
    if isinstance(session.virtualenv, nox.virtualenv.CondaEnv):
        print("using conda installer")
        conda_install(session, *args)
    else:
        print("using pip installer")
        session.install(*args)


def install_from_requirements(session: Session, *packages: str) -> None:
    """
    Install dependencies, respecting the version specified in requirements.
    """
    for package in packages:
        try:
            specs = REQUIRES["all"][package]
        except KeyError:
            raise ValueError(
                f"{package} cannot be found in {REQUIREMENT_PATH}."
            ) from None
        install(session, specs)


def install_extras(
    session: Session,
    extra: str = "core",
    force_pip: bool = False,
    pandas: str = "latest",
    pandas_stubs: bool = True,
) -> None:
    """Install dependencies."""
    specs, pip_specs = [], []
    pandas_version = "" if pandas == "latest" else f"=={pandas}"
    for spec in REQUIRES[extra].values():
        if spec == "pandas-stubs" and not pandas_stubs:
            # this is a temporary measure until all pandas-related mypy errors
            # are addressed
            continue

        req = Requirement(spec)  # type: ignore

        # this is needed until ray is supported on python 3.10
        # pylint: disable=line-too-long
        if req.name in {"ray", "geopandas"} and session.python == "3.10":  # type: ignore[attr-defined]  # noqa
            continue

        if req.name in ALWAYS_USE_PIP:  # type: ignore[attr-defined]
            pip_specs.append(spec)
        else:
            specs.append(
                spec if spec != "pandas" else f"pandas{pandas_version}"
            )
    if extra in {"core", "koalas", "modin", "fastapi"}:
        specs.append(REQUIRES["all"]["hypothesis"])

    # CI installs conda dependencies, so only run this for local runs
    if (
        isinstance(session.virtualenv, nox.virtualenv.CondaEnv)
        and not force_pip
        and not CI_RUN
    ):
        print("using conda installer")
        conda_install(session, *specs)
    else:
        print("using pip installer")
        session.install(*specs)

    # always use pip for these packages)
    session.install(*pip_specs)
    session.install("-e", ".", "--no-deps")  # install pandera


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
    install(session, "pyyaml")
    try:
        _generate_pip_deps_from_conda(session, compare=True)
    except nox.command.CommandFailed as err:
        _generate_pip_deps_from_conda(session)
        print(f"{REQUIREMENT_PATH} has been re-generated ✨ 🍰 ✨")
        raise err

    ignored_pkgs = {"black", "pandas"}
    mismatched = []
    # only compare package versions, not python version markers.
    str_dev_reqs = [str(x) for x in DEV_REQUIREMENTS]
    for extra, reqs in SETUP_REQUIREMENTS.items():
        for req in reqs:
            if (
                req.project_name not in ignored_pkgs
                and str(req) not in str_dev_reqs
            ):
                mismatched.append(f"{extra}: {req.project_name}")

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


@nox.session(python=DEFAULT_PYTHON)
def black(session: Session) -> None:
    """Check black style."""
    install_from_requirements(session, "black")
    args = ["--check"] if CI_RUN else session.posargs
    session.run(
        "black",
        f"--line-length={LINE_LENGTH}",
        *args,
        *SOURCE_PATHS,
    )


@nox.session(python=DEFAULT_PYTHON)
def isort(session: Session) -> None:
    """Check isort style."""
    install_from_requirements(session, "isort")
    args = ["--check-only"] if CI_RUN else session.posargs
    session.run(
        "isort",
        f"--line-length={LINE_LENGTH}",
        *args,
        *SOURCE_PATHS,
    )


@nox.session(python=PYTHON_VERSIONS)
def lint(session: Session) -> None:
    """Lint using pylint."""
    install_extras(session, extra="all")
    args = session.posargs or SOURCE_PATHS

    if session.python == "3.9":
        # https://github.com/PyCQA/pylint/issues/776
        args = ["--disable=unsubscriptable-object", *args]
    session.run("pylint", *args)


@nox.session(python=PYTHON_VERSIONS)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    install_extras(session, extra="all", pandas_stubs=False)
    args = session.posargs or SOURCE_PATHS
    session.run("mypy", "--follow-imports=silent", *args, silent=True)


EXTRA_NAMES = [
    extra
    for extra in REQUIRES
    if (
        extra != "all"
        and "python_version" not in extra
        and extra not in {"modin"}
    )
]


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pandas", PANDAS_VERSIONS)
@nox.parametrize("extra", EXTRA_NAMES)
def tests(session: Session, pandas: str, extra: str) -> None:
    """Run the test suite."""

    # skip these conditions
    python = (
        session.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    if (
        (pandas, extra)
        in {
            ("1.1.5", "koalas"),
            ("1.1.5", "modin-dask"),
            ("1.1.5", "modin-ray"),
        }
        or (python, pandas, extra)
        in {
            ("3.10", "1.1.5", "modin-dask"),
            ("3.10", "1.1.5", "modin-ray"),
        }
        or (python, extra)
        in {
            ("3.7", "modin-dask"),
            ("3.7", "modin-ray"),
            ("3.10", "modin-dask"),
            ("3.10", "modin-ray"),
            ("3.10", "koalas"),
        }
    ):
        session.skip()

    install_extras(session, extra, pandas=pandas)

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
            # strategies tests runs very slowly in python 3.7:
            # https://github.com/pandera-dev/pandera/issues/556
            # as a stop-gap, use the "dev" profile for 3.7
            profile = "ci" if CI_RUN and session.python != "3.7" else "dev"
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
    install_extras(session, extra="all", force_pip=True)
    session.run("xdoctest", PACKAGE, "--quiet")


@nox.session(python=PYTHON_VERSIONS)
def docs(session: Session) -> None:
    """Build the documentation."""
    # this is needed until ray and geopandas are supported on python 3.10
    if session.python == "3.10":
        session.skip()

    install_extras(session, extra="all", force_pip=True)
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
