.. _api-decisions:

Decision Parsing
================

Validate and parse unstructured data by asking a decision model, and judge
existing columns with semantic checks. See the
:ref:`decision parsing guide <decisions>` and the
:ref:`tutorial <decision-parsing-tutorial>` for how to use these objects.

.. currentmodule:: pandera.decisions

Question Types
--------------

A decision model answers typed questions; the question type says what shape
the answer has. These are the parser objects passed to ``ParsedField`` /
``ParsedColumn``.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Choice
   Score
   Noul
   Confidence

Providers
---------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DecisionProvider
   MockProvider
   RecordingProvider
   ReplayProvider

Configuration
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_provider
   get_provider
   provider
   register_provider
   capabilities_of
   enabled

Semantic Checks
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Holds
   holds

Caching and Inspection
----------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MemoryCache
   SQLiteCache
   Plan

.. autosummary::
   :toctree: generated
   :nosignatures:

   questions
   plan
   stats

Types and Errors
----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Decision
   ProviderCapabilities
   ProviderLimits
   DecisionsConfigError
   ProviderCapabilityError

``Question`` is a type alias (a union of the three question kinds) rather than
an instantiable class; it appears in the signatures of the objects above.
