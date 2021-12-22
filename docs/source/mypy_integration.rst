.. currentmodule:: pandera

.. _mypy_integration:

Mypy
====

*new in 0.8.0*

Pandera integrates with mypy to provide static type-linting of dataframes,
relying on `pandas-stubs <https://github.com/VirtusLab/pandas-stubs>`__
for typing information.

.. code:: bash

    pip install pandera[mypy]

Then enable the plugin in your ``mypy.ini`` or ``setug.cfg`` file:

.. code:: toml

    [mypy]
    plugins = pandera.mypy

.. note::

   Mypy static type-linting is supported for only pandas dataframes.

.. warning::

    This functionality is experimental 🧪. Since the
    `pandas-stubs <https://github.com/VirtusLab/pandas-stubs>`__ type stub
    annotations don't always match the official
    `pandas effort to support type annotations <https://github.com/pandas-dev/pandas/issues/28142#issuecomment-991967009>`__),
    installing the ```pandera[mypy]`` extra may yield false positives in your
    pandas code, many of which are are documented in ``tests/mypy/modules``.

    We encourage beta users to `file an issue <https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=bug,mypy&template=bug_report.md&title=>`__
    if they find any false positives or negatives being reported by ``mypy``.
    A list of such issues can be found `here <https://github.com/pandera-dev/pandera/labels/mypy>`__.


In the example below, we define a few schemas to see how type-linting with
pandera works.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 8-27

The mypy linter will complain if the output type of the function body doesn't
match the function's return signature.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 30-43

It'll also complain if the input type doesn't match the expected input type.
Note that we're using the :py:class:`pandera.typing.pandas.DataFrame` generic
type to define dataframes that are validated against the
:py:class:`~pandera.model.SchemaModel` type variable on initialization.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 47-60


To make mypy happy with respect to the return type, you can either initialize
a dataframe of the expected type:

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 63-64

.. note::
    If you use the approach above with the :py:func:`~pandera.check_types`
    decorator, pandera will do its best to not to validate the dataframe twice
    if it's already been initialized with the
    ``DataFrame[Schema](**data)`` syntax.

Or use :py:func:`typing.cast` to indicate to mypy that the return value of
the function is of the correct type.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 67-68


Limitations
^^^^^^^^^^^

An important caveat to static type-linting with pandera dataframe types is that,
since pandas dataframes are mutable objects, there's no way for ``mypy`` to
know whether a mutated instance of a
:py:class:`~pandera.model.SchemaModel`-typed dataframe has the correct
contents. Fortunately, we can simply rely on the :py:func:`~pandera.check_types`
decorator to verify that the output dataframe is valid.

Consider the examples below:

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 63-80

Even though the outputs of these functions are incorrect, mypy doesn't catch
the error during static type-linting but pandera will raise a
:py:class:`~pandera.errors.SchemaError` or :py:class:`~pandera.errors.SchemaErrors`
exception at runtime, depending on whether you're doing
:ref:`lazy validation<lazy_validation>` or not.

.. literalinclude:: ../../tests/core/static/pandas_dataframe.py
    :lines: 83-87
