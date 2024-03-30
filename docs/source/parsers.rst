.. pandera documentation for Parsers
.. currentmodule:: pandera

.. _parsers:

Parsers
=======

Parsers allow you to do some preprocessing on dataframes, columns, and series objects. Currently only support Pandas objects.
Both dataframe-level and component-level parsers are available and dataframe-level parsers will be performed first and component-level ones next.

Parsing column properties
-------------------------

:class:`~pandera.api.parsers.Parser` objects accept a function as a required argument, which is
expected to take a ``pa.Series`` input and output a parsed ``Series``, for example:


.. testcode:: parsers

    import pandera as pa
    import numpy as np
    import pandas as pd

    parser_sqrt = pa.Parser(lambda s: np.sqrt(s))

    schema = pa.DataFrameSchema({"column1": pa.Parser(float, parser_sqrt)})
    schema.validate(pd.DataFrame({"column1": [1., 2., 3.]}))


Multiple parsers can be applied to a column:

.. important::

    The order of ``parsers`` will be kept while parsing.


.. testcode:: parsers

    schema = pa.DataFrameSchema({
        "column2": pa.Column(str, parsers=[
            pa.Parser(lambda s: s.str.zfill(10)),
            pa.Parser(lambda s: s.str[2:]),
        ]),
    })

    schema.validate(pd.DataFrame({"column2": ["12345", "67890"]}))
