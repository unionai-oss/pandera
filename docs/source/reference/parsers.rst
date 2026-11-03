.. _api-parsers:

Parsers and Derived Columns
===========================

Preprocess data and declare columns that are derived from other columns. See
the :ref:`parsers guide <parsers>` for how to use these objects, and the
:ref:`decision parsing guide <decisions>` for deriving columns by asking a
decision model.

Parser
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.parsers.Parser
   pandera.api.parsers.ColumnParser
   pandera.api.parsers.ParseContext

Declared derivation on the column
---------------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.api.pandas.components.ParsedColumn

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.api.dataframe.model_components.ParsedField
   pandera.api.dataframe.model_components.ParsedFieldInfo
