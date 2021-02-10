# Contributing to pandera

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
changes to the codebase. Once you have obtained a copy of the code, you should
create a development environment that is separate from your existing Python
environment so that you can make and test changes without compromising your
own work environment.

An excellent guide on setting up python environments can be found
[here](https://pandas.pydata.org/docs/development/contributing.html#creating-a-python-environment).
Pandera offers a `environment.yml` to set up a conda-based environment and
`requirements-dev.txt` for a virtualenv.

### Contributing documentation

Maybe the easiest, fastest, and most useful way to contribute to this project
(and any other project) is to contribute documentation. If you find an API
within the project that doesn't have an example or description, or could be
clearer in its explanation, contribute yours!

This project uses Sphinx for auto-documentation and RST syntax for docstrings.
Once you have the code downloaded and you find something that is in need of some
TLD, take a look at the [Sphinx](https://www.sphinx-doc.org/en/1.0/rest.html)
documentation or well-documented [examples](https://pandera.readthedocs.io/en/stable/_modules/pandera/schemas.html#DataFrameSchema)
within the codebase for guidance on contributing.

You can build the html documentation by running `nox -s docs`.


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

### Set up `pre-commit`

This project uses [pre-commit](https://pre-commit.com/) to ensure that code
standard checks pass locally before pushing to the remote project repo. Follow
the [installation instructions](https://pre-commit.com/#installation), then
set up hooks with `pre-commit install`. After, `black`, `pylint` and `mypy` checks should
be run with every commit.

### Run the test suite

Before submitting your changes for review, make sure to check that your changes
do not break any tests by running: `nox` or `nox -db conda` depending on your environment.


### Raising Pull Requests

Once your changes are ready to be submitted, make sure to push your changes to
your fork of the GitHub repo before creating a pull request. We will review
your changes, and might ask you to make additional changes before it is finally
ready to merge. However, once it's ready, we will merge it, and you will have
successfully contributed to the codebase!
