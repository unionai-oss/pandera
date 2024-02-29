Granular Dataframe Validations
==============================

By default, error reports are generated for both schema and data level validation,
but more granular control over schema or data only validations is available.

This is achieved by introducing configurable settings using environment variables
that allow you to control execution at three different levels:

1. ``SCHEMA_ONLY``: perform schema validations only. It checks that data conforms
   to the schema definition, but does not perform any data-level validations on dataframe.
2. ``DATA_ONLY``: perform data-level validations only. It validates that data
   conforms to the defined ``checks``, but does not validate the schema.
3. ``SCHEMA_AND_DATA``: (**default**) perform both schema and data level
   validations. It runs most exhaustive validation and could be compute intensive.

You can override default behaviour by setting an environment variable from terminal
before running the ``pandera`` process as:

.. code-block:: bash

    export PANDERA_VALIDATION_DEPTH=SCHEMA_ONLY

This will be picked up by ``pandera`` to only enforce SCHEMA level validations.
