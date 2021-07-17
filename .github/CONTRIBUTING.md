# Contributing

Whether you are a novice or experienced software developer, all contributions
and suggestions are welcome!

## Getting Started

If you are looking to contribute to the _pandera_ codebase, the best place to
start is the [GitHub "issues" tab](https://github.com/pandera-dev/pandera/issues).
This is also a great place for filing bug reports and making suggestions for
ways in which we can improve the code and documentation.

## Contributing to the Codebase

The code is hosted on [GitHub](https://github.com/pandera-dev/pandera/issues),
so you will need to use [Git](http://git-scm.com/) to clone the project and make
changes to the codebase.

First create your own fork of pandera, then clone it:

```
# replace <my-username> with your github username
git clone https://github.com/<my-username>/pandera.git
```

Once you've obtained a copy of the code, create a development environment that's
separate from your existing Python environment so that you can make and test
changes without compromising your own work environment.

An excellent guide on setting up python environments can be found
[here](https://pandas.pydata.org/docs/development/contributing.html#creating-a-python-environment).
Pandera offers a `environment.yml` to set up a conda-based environment and
`requirements-dev.txt` for a virtualenv.

### Environment Setup

#### Option 1: `miniconda` Setup

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), then run:

```bash
conda create -n pandera-dev python=3.8  # or any python version 3.7+
conda env update -n pandera-dev -f environment.yml
conda activate pandera-dev
pip install -e .
```

#### Option 2: `virtualenv` Setup

```bash
pip install virtualenv
virtualenv .venv/pandera-dev
pip install -r requirements-dev.txt
pip install -e .
```

#### Run Tests

```
pytest tests
```

#### Set up `pre-commit`

This project uses [pre-commit](https://pre-commit.com/) to ensure that code
standard checks pass locally before pushing to the remote project repo. Follow
the [installation instructions](https://pre-commit.com/#installation), then
set up hooks with `pre-commit install`. After, `black`, `pylint` and `mypy`
checks should be run with every commit.

Make sure everything is working correctly by running

```
pre-commit run --all
```

### Making Changes

Before making changes to the codebase or documentation, create a new branch with:

```
git checkout -b <my-branch>
```

We recommend following the branch-naming convention described in [Making Pull Requests](#making-pull-requests).

### Run the Full Test Suite Locally

Before submitting your changes for review, make sure to check that your changes
do not break any tests by running:

```
# option 1: if you're working with conda (recommended)
$ make nox-conda

# option 2: if you're working with virtualenv
$ make nox
```

Option 2 assumes that you have python environments for all of the versions
that pandera supports.

#### Using `mamba` (optional)

You can also use [mamba](https://github.com/mamba-org/mamba), which is a faster
implementation of [miniconda](https://docs.conda.io/en/latest/miniconda.html),
to run the `nox` test suite. Simply install it via conda-forge, and
`make nox-conda` should use it under the hood.

```
$ conda install -c conda-forge mamba
$ make nox-conda
```

### Project Releases

Releases are organized under [milestones](https://github.com/pandera-dev/pandera/milestones),
which are be associated with a corresponding branch. This project uses
[semantic versioning](https://semver.org/), and we recommend prioritizing issues
associated with the next release.

### Contributing Documentation

Maybe the easiest, fastest, and most useful way to contribute to this project
(and any other project) is to contribute documentation. If you find an API
within the project that doesn't have an example or description, or could be
clearer in its explanation, contribute yours!

You can also find issues for improving documentation under the
[docs](https://github.com/pandera-dev/pandera/labels/docs) label. If you have
ideas for documentation improvements, you can create a new issue [here](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=docs&template=documentation-improvement.md&title=)

This project uses Sphinx for auto-documentation and RST syntax for docstrings.
Once you have the code downloaded and you find something that is in need of some
TLD, take a look at the [Sphinx](https://www.sphinx-doc.org/en/1.0/rest.html)
documentation or well-documented
[examples](https://pandera.readthedocs.io/en/stable/_modules/pandera/schemas.html#DataFrameSchema)
within the codebase for guidance on contributing.

You can build the html documentation by running `nox -s docs`. The built
documentation can be found in `docs/_build`.

### Contributing Bugfixes

Bugs are reported under the [bug](https://github.com/pandera-dev/pandera/labels/bug)
label, so if you find a bug create a new issue [here](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=bug&template=bug_report.md&title=).

### Contributing Enhancements

New feature issues can be found under the
[enhancements](https://github.com/pandera-dev/pandera/labels/enhancement) label.
You can request a feature by creating a new issue [here](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=).
### Making Pull Requests

Once your changes are ready to be submitted, make sure to push your changes to
your fork of the GitHub repo before creating a pull request. Depending on the
type of issue the pull request is resolving, your pull request should merge
onto the appropriate branch:

#### Bugfixes
- branch naming convention: `bugfix/<issue number>` or `bugfix/<bugfix-name>`
- pull request to: `dev`

#### Documentation
- branch naming convention: `docs/<issue number>` or `docs/<doc-name>`
- pull request to: `release/x.x.x` branch if specified in the issue milestone, otherwise `dev`

#### Enhancements
- branch naming convention: `feature/<issue number>` or `feature/<bugfix-name>`
- pull request to: `release/x.x.x` branch if specified in the issue milestone, otherwise `dev`

We will review your changes, and might ask you to make additional changes
before it is finally ready to merge. However, once it's ready, we will merge
it, and you will have successfully contributed to the codebase!

### Questions, Ideas, General Discussion

Head on over to the [discussion](https://github.com/pandera-dev/pandera/discussions)
section if you have questions or ideas, want to show off something that you
did with `pandera`, or want to discuss a topic related to the project.

### Dataframe Schema Style Guides

We have guidelines regarding dataframe and schema styles that are encouraged
for each pull request:

- If specifying a single column DataFrame, this can be expressed as a one-liner:

  ```python
  DataFrameSchema({"col1": Column(...)})
  ```

- If specifying one column with multiple lines, or multiple columns:

  ```python
  DataFrameSchema(
      {
          "col1": Column(
              int,
              checks=[
                  Check(...),
                  Check(...),
              ]
          ),
      }
  )
  ```

- If specifying columns with additional arguments that fit in one line:

  ```python
  DataFrameSchema(
      {"a": Column(int, nullable=True)},
      strict=True
  )
  ```

- If specifying columns with additional arguments that don't fit in one line:
  ```python
  DataFrameSchema(
      {
          "a": Column(
              int,
              nullable=True,
              coerce=True,
              ...
          ),
          "b": Column(
              ...,
          )
      },
      strict=True)
  ```
