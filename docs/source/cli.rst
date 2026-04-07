.. _cli:
.. _api-cli:

CLI
===

The Pandera CLI can **validate** on-disk datasets against a serialized schema
(YAML or JSON), **infer** a schema from data and write YAML, JSON, or a Python
module, or **generate** synthetic tabular or xarray data from a pandas or
xarray schema. It is useful in scripts and CI without writing Python glue code.

Installation
----------

The CLI depends on `Typer <https://typer.tiangolo.com/>`__. Install Pandera with
the ``cli`` extra:

.. code-block:: bash

   pip install 'pandera[cli]'

You still need the appropriate dataframe library for the schema you validate
(for example ``pandas`` for the default pandas-API schema). YAML schemas require
``pandera[io]`` (PyYAML).

Refer to :doc:`cli_guide` to learn how to use the CLI.

Commands
--------

.. typer:: pandera._cli.app
    :prog: pandera
    :markup-mode: markdown
    :theme: dimmed_monokai
    :show-nested:
    :make-sections:
    :width: 79
    :preferred: svg

Supported file combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schema files must be YAML (with PyYAML installed) or JSON. Supported data
extensions depend on the backend; common cases include CSV, Parquet, JSON, and
Feather for pandas-like backends. Ibis is limited to CSV and Parquet in the CLI
(see the implementation for details).

For more on serialization formats, see :ref:`Schema persistence <schema-persistence>`
and :ref:`IO Utilities <api-io-utils>`.
