# Contributing to pandera

Whether you are a novice or experienced software developer, all contributions
and suggestions are welcome!

## Getting Started

If you are looking to contribute to the *pandera* codebase, the best place to
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

### Dataframe Style Guides
We have guidelines regarding dataframe and schema styles that are enforced for
each pull request:

- If specifying a single column DataFrame, this can be expressed as a one-liner:
```DataFrameSchema({"col1": Column(...)})```

- If specifying one column with multiple lines, or multiple columns:
    ```
    DataFrameSchema({
        "col1": Column(type, checks=[
            Check(...),
            Check(...),
        ]),
    })


    DataFrameSchema({
        "col1": Column(...),
        "col2": Column(...),
    })
    ```

- If specifying single columns with additional arguments
    ```
    DataFrameSchema({"a": Column(Int, nullable=True)},
                    strict=True)
    ```

- If specifying columns with additional arguments
    ```
    DataFrameSchema(
        {
            "col1": Column(...),
            "col2": Column(...),
        },
        strict=True)
    ```

### Run the tests
Before submitting your changes for review, make sure to check that your changes
do not break any tests by running: ``pytest tests/``

### Raising Pull Requests

Once your changes are ready to be submitted, make sure to push your changes to
your fork of the GitHub repo before creating a pull request.  We will review
your changes, and might ask you to make additional changes before it is finally
ready to merge. However, once it's ready, we will merge it, and you will have
successfully contributed to the codebase!
