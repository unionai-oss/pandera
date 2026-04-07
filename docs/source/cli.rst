.. _cli:
.. _api-cli:

CLI
===

The Pandera CLI validates on-disk datasets against a serialized schema (YAML or
JSON). It is useful in scripts and CI without writing Python glue code.

Installation
----------

The CLI depends on `Typer <https://typer.tiangolo.com/>`__. Install Pandera with
the ``cli`` extra:

.. code-block:: bash

   pip install 'pandera[cli]'

You still need the appropriate dataframe library for the schema you validate
(for example ``pandas`` for the default pandas-API schema). YAML schemas require
``pandera[io]`` (PyYAML).

``validate`` subcommand
-----------------------

Run validation with the ``validate`` subcommand, passing a schema path and a data
file path:

.. code-block:: bash

   pandera validate --schema path/to/schema.yaml --data path/to/data.csv

Short form:

.. code-block:: bash

   pandera validate -s schema.yaml -d data.csv

The same entry point works with ``python -m pandera``:

.. code-block:: bash

   python -m pandera validate -s schema.yaml -d data.csv

On success the process exits with code ``0`` and prints
``Validation succeeded.`` On validation failure it exits with a non-zero code
and prints an error message.

Options
~~~~~~~

.. list-table::
   :widths: 18 12 70
   :header-rows: 1

   * - Option
     - Shorthand
     - Description
   * - ``--schema``
     - ``-s``
     - Path to a ``.yaml``, ``.yml``, or ``.json`` schema file.
   * - ``--data``
     - ``-d``
     - Path to the dataset; allowed extensions depend on the inferred backend
       (see below).
   * - ``--backend``
     - ``-b``
     - Optional. Force a dataframe library (``pandas``, ``polars``, ``dask``,
       ``modin``, ``pyspark.pandas``, ``pyspark.sql``, ``ibis``). Must match what
       the schema implies; if omitted, the backend is inferred from the schema
       file.

Backend inference
~~~~~~~~~~~~~~~~~~~

The CLI reads ``schema_type`` and, for pandas-API schemas,
``dataframe_library`` from the serialized schema to decide which backend to
use—for example Polars when ``schema_type`` is ``polars_dataframe``, or Ibis
when it is ``ibis_table``. Use ``--backend`` only when you want to be explicit;
it must agree with the schema metadata.

Full example (pandas CSV + JSON schema)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet below installs what you need, writes a small CSV and a matching JSON
schema (via :func:`pandera.io.pandas_io.to_json`), then runs the CLI. A JSON
schema avoids needing PyYAML; for YAML schemas, install ``pandera[io]`` as well.

Run the whole block in a shell from an empty directory (or change the file
names):

.. code-block:: bash

   pip install 'pandera[pandas,cli]'

.. code-block:: bash

   python <<'PY'
   import pandas as pd
   import pandera.pandas as pa
   from pandera.io import pandas_io as io

   pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}).to_csv(
       "sample_data.csv",
       index=False,
   )
   schema = pa.DataFrameSchema(
       {
           "id": pa.Column(int),
           "name": pa.Column(str),
       }
   )
   io.to_json(schema, "sample_schema.json")
   PY

.. code-block:: bash

   pandera validate -s sample_schema.json -d sample_data.csv

You should see ``Validation succeeded.`` and exit code ``0``.

If your shell does not support heredocs (for example ``cmd.exe`` on Windows), save
the Python block to a file and run ``python create_sample_files.py``, then run
the ``pandera validate`` line.

Supported file combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schema files must be YAML (with PyYAML installed) or JSON. Supported data
extensions depend on the backend; common cases include CSV, Parquet, JSON, and
Feather for pandas-like backends. Ibis is limited to CSV and Parquet in the CLI
(see the implementation for details).

For more on serialization formats, see :ref:`Schema persistence <schema-persistence>`
and :ref:`IO Utilities <api-io-utils>`.

Generated command help
----------------------

The command tree below is generated from the Typer application with
`sphinxcontrib-typer <https://github.com/sphinx-contrib/typer>`_ (see
`its documentation <https://sphinxcontrib-typer.readthedocs.io/>`__).
The application object is ``pandera._cli.app`` (it is not rendered with
autodoc because inherited Typer docstrings confuse Sphinx).
HTML output uses a fixed ``iframe-height`` so builds do not need Selenium
(dynamic height would otherwise require ``selenium`` and ``webdriver-manager``,
see `typer_get_iframe_height
<https://sphinxcontrib-typer.readthedocs.io/en/latest/reference/configuration.html>`__).

.. typer:: pandera._cli.app
    :prog: pandera
    :show-nested:
    :make-sections:
    :width: 79
    :preferred: text
