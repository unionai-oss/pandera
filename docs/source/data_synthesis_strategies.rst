.. pandera documentation for synthesizing data

.. currentmodule:: pandera

.. _data synthesis strategies:

Data Synthesis Strategies
=========================

*new in 0.6.0*

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
:class:`~pandera.core.checks.Check` s  in order to make the data generation process
deterministic for documentation purposes.

Usage in Unit Tests
-------------------

The ``example`` method is available for all schemas and schema components, and
is primarily meant to be used interactively. It *could* be used in a script to
generate test cases, but ``hypothesis`` recommends against doing this and
instead using the ``strategy`` method to create a ``hypothesis`` strategy
that can be used in ``pytest`` unit tests.

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
a ``strategy`` that can then be collected by a `pytest <https://docs.pytest.org/en/latest/>`__
runner. We could also run the tests explicitly ourselves, or run it as a
``unittest.TestCase``. For more information on testing with hypothesis, see the
`hypothesis quick start guide <https://hypothesis.readthedocs.io/en/latest/quickstart.html#running-tests>`__.

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

Strategies and Examples from DataFrame Models
---------------------------------------------

You can also use the :ref:`class-based API<dataframe_models>` to generate examples.
Here's the equivalent dataframe model for the above examples:

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   from pandera.typing import Series, DataFrame

   class InSchema(pa.DataFrameModel):
       column1: Series[int] = pa.Field(eq=10)
       column2: Series[float] = pa.Field(eq=0.25)
       column3: Series[str] = pa.Field(eq="foo")

   class OutSchema(InSchema):
       column4: Series[float]

   @pa.check_types
   def processing_fn(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
       return df.assign(column4=df.column1 * df.column2)

   @hypothesis.given(InSchema.strategy(size=5))
   def test_processing_fn(dataframe):
       processing_fn(dataframe)


Checks as Constraints
---------------------

As you may have noticed in the first example, :class:`~pandera.core.checks.Check` s
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

   for _ in range(5):
       # generate 10 rows of the dataframe
       sample_data = schema_multiple_checks.example(size=3)

       # validate the sampled data
       schema_multiple_checks(sample_data)

One caveat here is that it's up to you to define a set of checks that are
jointly satisfiable. If not, an ``Unsatisfiable`` exception will be raised:

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   schema_multiple_checks = pa.DataFrameSchema({
       "column1": pa.Column(
           float, checks=[
               # nonsensical constraints
               pa.Check.gt(0),
               pa.Check.lt(-10),
           ]
        )
   })

   schema_multiple_checks.example(size=3)

.. testoutput:: data_synthesis_strategies

    Traceback (most recent call last):
    ...
    Unsatisfiable: Unable to satisfy assumptions of hypothesis example_generating_inner_function.

.. _check strategy chaining:

Check Strategy Chaining
~~~~~~~~~~~~~~~~~~~~~~~

If you specify multiple checks for a particular column, this is what happens
under the hood:

- The first check in the list is the *base strategy*, which ``hypothesis``
  uses to generate data.
- All subsequent checks filter the values generated by the previous strategy
  such that it fulfills the constraints of current check.

To optimize efficiency of the data-generation procedure, make sure to specify
the most restrictive constraint of a column as the *base strategy* and build
other constraints on top of it.

In-line Custom Checks
~~~~~~~~~~~~~~~~~~~~~

One of the strengths of ``pandera`` is its flexibility with regard to defining
custom checks on the fly:

.. testcode:: data_synthesis_strategies
   :skipif: SKIP_STRATEGY

   schema_inline_check = pa.DataFrameSchema({
       "col": pa.Column(str, pa.Check(lambda s: s.isin({"foo", "bar"})))
   })


One of the disadvantages of this is that the fallback strategy is to simply
apply the check to the generated data, which can be highly inefficient. In this
case, ``hypothesis`` will generate strings and try to find examples of strings
that are in the set ``{"foo", "bar"}``, which will be very slow and most likely
raise an ``Unsatisfiable`` exception. To get around this limitation, you can
register custom checks and define strategies that correspond to them.


Defining Custom Strategies
--------------------------

All built-in :class:`~pandera.core.checks.Check` s are associated with a data
synthesis strategy. You can define your own data synthesis strategies by using
the :ref:`extensions API<extensions>` to register a custom check function with
a corresponding strategy.
