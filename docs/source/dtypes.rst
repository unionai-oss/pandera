.. pandera documentation for check_input and check_output decorators

.. currentmodule:: pandera

.. _dtypes:

Pandera Data Types
==================

*new in 0.7.0*

.. _dtypes-into:

Motivations
~~~~~~~~~~~

Pandera defines its own interface for data types in order to abstract the
specifics of dataframe-like data structures in the python ecosystem, such
as Apache Spark, Apache Arrow and xarray.

.. note:: In the following section ``Pandera Data Type`` refers to a
    :class:`pandera.dtypes.DataType` object whereas ``native data type`` refers
    to data types used by third-party libraries that Pandera supports (e.g. pandas).

Most of the time, it is transparent to end users since pandera columns and
indexes accept native data types. However, it is possible to extend the pandera
interface by:

* modifying the **data type check** performed during schema validation.
* modifying the behavior of the **coerce** argument for :class:`~pandea.schemas.DataFrameSchema`.
* adding your **own custom data types**.

DataType basics
~~~~~~~~~~~~~~~

All pandera data types inherit from :class:`pandera.dtypes.DataType` and must
be hashable.

A data type implements three key methods:

* :meth:`pandera.dtypes.DataType.check` which validates that data types are equivalent.
* :meth:`pandera.dtypes.DataType.coerce` which coerces a data container
  (e.g. :class:`pandas.Series`) to the data type.
* The dunder method ``__str__()`` which should output the native alias.
  For example ``str(pandera.Float64) == "float64"``


For pandera's validation methods to be aware of a data type, it has to be
registered with the targeted engine via :meth:`pandera.engines.engine.Engine.register_dtype`.
An engine is in charge of mapping a pandera :class:`~pandera.dtypes.DataType`
with a native data type counterpart belonging to a third-party library. The mapping
can be queried with :meth:`pandera.engines.engine.Engine.dtype`.

As of pandera ``0.7.0``, only the pandas :class:`~pandera.engines.pandas_engine.Engine`
is supported.


Example
~~~~~~~

Let's extend :class:`pandas.BooleanDtype` coercion to handle the string
literals ``"True"`` and ``"False"``.

.. testcode:: dtypes

    import pandas as pd
    import pandera as pa
    from pandera import dtypes
    from pandera.engines import pandas_engine


    @pandas_engine.Engine.register_dtype  # step 1
    @dtypes.immutable  # step 2
    class LiteralBool(pandas_engine.BOOL):  # step 3
        def coerce(self, series: pd.Series) -> pd.Series:
            """Coerce a pandas.Series to date types."""
            if pd.api.types.is_string_dtype(series):
                series = series.replace({"True": 1, "False": 0})
            return series.astype("boolean")


    data = pd.Series(["True", "False"], name="literal_bools")

    # step 4
    print(
        pa.SeriesSchema(LiteralBool(), coerce=True, name="literal_bools")
        .validate(data)
        .dtype
    )

.. testoutput:: dtypes

   boolean

The example above performs the following steps:

1. Register the data type with the pandas engine.
2. :func:`pandera.dtypes.immutable` creates an immutable (and hashable)
   :func:`dataclass`.
3. Inherit :class:`pandera.engines.pandas_engine.BOOL`, which is the pandera
   representation of :class:`pandas.BooleanDtype`. This is not mandatory but
   it makes our life easier by having already implemented all the required
   methods.
4. Check that our new data type can coerce the string literals.

So far we did not override the default behavior:

.. testcode:: dtypes

    import pandera as pa

    pa.SeriesSchema("boolean", coerce=True).validate(data)


.. testoutput:: dtypes

    Traceback (most recent call last):
    ...
    pandera.errors.SchemaError: Error while coercing 'literal_bools' to type boolean: Need to pass bool-like values

To completely replace the default :class:`~pandera.engines.pandas_engine.BOOL`,
we need to supply all the equivalent representations to
:meth:`~pandera.engines.engine.Engine.register_dtype`. Behind the scenes, when
``pa.SeriesSchema("boolean")`` is called the corresponding pandera data type
is looked up using :meth:`pandera.engines.engine.Engine.dtype`.

.. testcode:: dtypes

    print(f"before: {pandas_engine.Engine.dtype('boolean').__class__}")


    @pandas_engine.Engine.register_dtype(
        equivalents=["boolean", pd.BooleanDtype, pd.BooleanDtype()],
    )
    @dtypes.immutable
    class LiteralBool(pandas_engine.BOOL):
        def coerce(self, series: pd.Series) -> pd.Series:
            """Coerce a pandas.Series to date types."""
            if pd.api.types.is_string_dtype(series):
                series = series.replace({"True": 1, "False": 0})
            return series.astype("boolean")


    print(f"after: {pandas_engine.Engine.dtype('boolean').__class__}")

    for dtype in ["boolean", pd.BooleanDtype, pd.BooleanDtype()]:
        pa.SeriesSchema(dtype, coerce=True).validate(data)

.. testoutput:: dtypes

    before: <class 'pandera.engines.pandas_engine.BOOL'>
    after: <class 'LiteralBool'>

.. note:: For convenience, we specified both ``pd.BooleanDtype`` and
    ``pd.BooleanDtype()`` as equivalents. That gives us more flexibility in
    what pandera schemas can recognize (see last for-loop above).

Parametrized data types
~~~~~~~~~~~~~~~~~~~~~~~

Some data types can be parametrized. One common example is
:class:`pandas.CategoricalDtype`.

The ``equivalents`` argument of
:meth:`~pandera.engines.engine.Engine.register_dtype` does not handle
this situation but will automatically register a :func:`classmethod` with
signature ``from_parametrized_dtype(cls, equivalent:...)`` if the decorated
:class:`~pandera.dtypes.DataType` defines it. The ``equivalent`` argument must
be type-annotated because it is leveraged to dispatch the input of
:class:`~pandera.engines.engine.Engine.dtype` to the appropriate
``from_parametrized_dtype`` class method.

For example, here is a snippet from :class:`pandera.engines.pandas_engine.Category`:

.. code-block:: python

    import pandas as pd
    from pandera import dtypes

    @classmethod
    def from_parametrized_dtype(
        cls, cat: Union[dtypes.Category, pd.CategoricalDtype]
    ):
        """Convert a categorical to
        a Pandera :class:`pandera.dtypes.pandas_engine.Category`."""
        return cls(categories=cat.categories, ordered=cat.ordered)  # type: ignore


.. note:: The dispatch mechanism relies on :func:`functools.singledispatch`.
    Unlike the built-in implementation, :data:`typing.Union` is recognized.


Defining the ``coerce_value`` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For pandera datatypes to understand how to correctly report coercion errors,
it needs to know how to coerce an individual value into the specified type.

All ``pandas`` data types are supported: ``numpy`` -based datatypes use the
underlying numpy dtype to coerce an individual value. The ``pandas`` -native
datatypes like :class:`~pandas.CategoricalDtype` and :class:`~pandas.BooleanDtype`
are also supported.

As an example of a special-cased ``coerce_value`` implementation, see
:py:meth:`~pandera.engines.pandas_engine.Category.coerce_value`:


.. literalinclude:: ../../pandera/engines/pandas_engine.py
   :lines: 580-586

And :py:meth:`~pandera.engines.pandas_engine.BOOL.coerce_value`:

.. literalinclude:: ../../pandera/engines/pandas_engine.py
   :lines: 223-229

Logical data types
~~~~~~~~~~~~~~~~~~

Taking inspiration from the `visions project <https://dylan-profiler.github.io/visions/visions/background/data_type_view.html#decoupling-physical-and-logical-types>`_,
pandera provides an interface for defining logical data types.

Physical types represent the actual, underlying representation of the data.
e.g.: ``Int8``, ``Float32``, ``String``, etc., whereas logical types represent the
abstracted understanding of that data. e.g.: ``IPs``, ``URLs``, ``paths``, etc.

Validating a logical data type consists of validating the supporting physical data type
(see :ref:`dtypes-into`) and a check on actual values. For example, an IP address data
type would validate that:

1. The data container type is a ``String``.
2. The actual values are well-formed addresses.

Non-native Pandas dtype can also be wrapped in a :class:`numpy.object_` and verified
using the data, since the `object` dtype alone is not enough to verify the
correctness. An example would be the standard :class:`decimal.Decimal` class that can be
validated via the pandera DataType :class:`~pandera.dtypes.Decimal`.

To implement a logical data type, you just need to implement the method
:meth:`pandera.dtypes.DataType.check` and make use of the ``data_container`` argument to
perform checks on the values of the data.
