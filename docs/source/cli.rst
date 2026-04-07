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

Commands
--------

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

``infer`` subcommand
~~~~~~~~~~~~~~~~~~~~

Infer a schema from a dataset using the same file extensions as ``validate``
for the chosen ``--backend``. The output format is taken from the output path
(``.yaml``/``.yml``, ``.json``, or ``.py``), or set explicitly with
``--format``. For ``.py`` output, use ``--script-type schema`` (default) or
``model``; formatting requires `black <https://black.readthedocs.io/>`__.

.. code-block:: bash

   pandera infer -d sample_data.csv -o sample_schema.yaml

``generate`` subcommand
~~~~~~~~~~~~~~~~~~~~~~~

Produce example data that satisfies a serialized schema using `hypothesis
<https://hypothesis.readthedocs.io/>`__ strategies (install ``pandera[strategies]``).
Only **pandas** dataframe schemas and **xarray** ``data_array`` / ``dataset``
schemas are supported for now.

* **Pandas**: write ``.csv``, ``.json``, ``.parquet``, or ``.feather``.
* **Xarray**: write NetCDF (``.nc``) or, when the object can be represented as
  a table, the same tabular formats as above.

.. code-block:: bash

   pandera generate -s sample_schema.yaml -o synthetic.csv

Backend inference
~~~~~~~~~~~~~~~~~~~

The CLI reads ``schema_type`` and, for pandas-API schemas,
``dataframe_library`` from the serialized schema to decide which backend to
use—for example Polars when ``schema_type`` is ``polars_dataframe``, or Ibis
when it is ``ibis_table``. Use ``--backend`` only when you want to be explicit;
it must agree with the schema metadata.

Full example: pandas CSV + JSON schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
           "id": pa.Column(int, pa.Check.ge(0)),
           "name": pa.Column(str, pa.Check.isin(["a", "b", "c"])),
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
