.. pandera documentation for synthesizing data

.. currentmodule:: pandera

.. _data synthesis strategies:

Data Synthesis Strategies (new)
===============================

*new in 0.6.0*

.. warning::

   This functionality is experimental. The API and behavior is considered
   unstable and may change at any time.

``pandera`` provides a utility for generating synthetic data purely from
pandera schema or schema component objects. Under the hood, the schema metadata
is collected to create a data-generating strategy using
`hypothesis <https://hypothesis.readthedocs.io/en/latest/>`__, which is a
property-based testing library.


Basic Usage
-----------

Once you've defined a schema, it's easy to generate examples:

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   import pandera as pa

   schema = pa.DataFrameSchema(
       {
           "column1": pa.Column(int, pa.Check.eq(10)),
           "column2": pa.Column(float, pa.Check.eq(0.25)),
           "column3": pa.Column(str, pa.Check.eq("foo")),
       }
   )
   print(schema.example(size=3))

.. testoutput:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

       column1  column2 column3
    0       10     0.25     foo
    1       10     0.25     foo
    2       10     0.25     foo


Note that here we've constrained the specific values in each column using
:class:`~pandera.checks.Check` s  in order to make the data generation process
deterministic for documentation purposes.

Pytest Usage
------------

The ``example`` method is available for all schemas and schema components, and
is primarily meant to be used interactively. It *could* be used in a script to
generate test cases, but `hypothesis <https://hypothesis.readthedocs.io/en/latest/>`__
recommends against doing this and instead using the ``strategy`` method and
to create a ``hypothesis`` strategy that can be used in ``pytest`` unit tests.

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   import hypothesis

   def processing_fn(df):
       return df.assign(column4=df.column1 * df.column2)

   @hypothesis.given(schema.strategy(size=5))
   def test_processing_fn(dataframe):
       result = processing_fn(dataframe)
       assert "column4" in result


The above example is trivial, but you get the idea! Schema objects can create
a ``strategy`` that can then be used by a `pytest <https://docs.pytest.org/en/latest/>`__
function using the ``hypothesis`` platform. For more information on strategies,
see the `hypothesis quick start guide <https://hypothesis.readthedocs.io/en/latest/quickstart.html#>`__.

A more practical example involves using
:ref:`schema transformations<dataframe schema transformations>`. We can modify
the function above to make sure that ``processing_fn`` actually outputs the
correct result:

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   out_schema = schema.add_columns({"column4": pa.Column(float)})

   @pa.check_output(out_schema)
   def processing_fn(df):
       return df.assign(column4=df.column1 * df.column2)

   @hypothesis.given(schema.strategy(size=5))
   def test_processing_fn(dataframe):
       processing_fn(dataframe)

Now the ``test_processing_fn`` simply becomes an execution test, raising a
:class:`~pandera.errors.SchemaError` if ``processing_fn`` doesn't add
``column4`` to the dataframe.


Checks as Constraints
---------------------

As you may have noticed in the first example, :class:`~pandera.checks.Check` s
further constrain the data synthesized from a strategy. Without checks, the
``example`` method would simply generate any value of the specified type. You
can specify multiple checks on a column and ``pandera`` should be able to
generate valid data under those constraints.

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   schema_multiple_checks = pa.DataFrameSchema({
       "column1": pa.Column(
           float, checks=[
               pa.Check.gt(0),
               pa.Check.lt(1e10),
               pa.Check.notin([-100, -10, 0]),
           ]
        )
   })

   for _ in range(100):
       # generate 10 rows of the dataframe
       sample_data = schema_multiple_checks.example(size=10)

       # validate the sampled data
       schema_multiple_checks(sample_data)

One caveat here is that it's up to you to define a set of checks that are
jointly satisfiable. If not, an error will be raised:

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

Check Strategy Chaining
-----------------------

Defining Custom Strategies
--------------------------
