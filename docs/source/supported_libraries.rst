.. currentmodule:: pandera

.. _supported-dataframe-libraries:

Supported DataFrame Libraries (New)
===================================

Pandera started out as a pandas-specific dataframe validation library, and
moving forward its core functionality will continue to support pandas. However,
pandera's adoption has resulted in the realization that it can be a much more
powerful tool by supporting other dataframe-like formats.

Scaling Up Data Validation
--------------------------

Pandera provides multiple ways of scaling up data validation to dataframes
that don't fit into memory. Fortunately, pandera doesn't have to re-invent
the wheel. Standing on shoulders of giants, it integrates with the existing
ecosystem of libraries that allow you to perform validations on out-of-memory
dataframes.

.. list-table::
   :widths: 25 75

   * - :ref:`Dask <scaling_dask>`
     - Apply pandera schemas to Dask dataframe partitions.
   * - :ref:`Fugue <scaling_fugue>`
     - Apply pandera schemas to distributed dataframe partitions with Fugue.
   * - :ref:`Koalas <scaling_koalas>`
     - A pandas drop-in replacement, distributed using a Spark backend.
   * - :ref:`Modin <scaling_modin>`
     - A pandas drop-in replacement, distributed using a Ray or Dask backend.
   * - :ref:`GeoPandas <scaling_geopandas>`
     - An extension of pandas that supports geospatial data.

.. toctree::
    :maxdepth: 1
    :caption: Introduction
    :hidden:

    Dask <dask>
    Fugue <fugue>
    Koalas <koalas>
    Modin <modin>

.. note::

   Don't see a library that you want supported? Check out the
   `github issues <https://github.com/pandera-dev/pandera/issues>`__ to see if
   that library is in the roadmap. If it isn't, open up a
   `new issue <https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=>`__
   to add support for it!
