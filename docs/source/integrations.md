(integrations)=

# Integrations

Pandera ships with integrations with other tools in the Python ecosystem, with
the goal of interoperating with libraries that you know and love.

```{eval-rst}
.. list-table::
   :widths: 25 75

   * - :ref:`FastAPI <fastapi-integration>`
     - Use pandera DataFrameModels in your FastAPI app
   * - :ref:`Frictionless <frictionless-integration>`
     - Convert frictionless schemas to pandera schemas
   * - :ref:`Hypothesis <data-synthesis-strategies>`
     - Use the hypothesis library to generate valid data under your schema's constraints.
   * - :ref:`Mypy <mypy-integration>`
     - Type-lint your pandas and pandera code with mypy for static type safety [experimental 🧪]
   * - :ref:`Pydantic <pydantic-integration>`
     - Use pandera DataFrameModels when defining  your pydantic BaseModels
```

```{toctree}
:caption: Introduction
:hidden: true
:maxdepth: 1

FastAPI <fastapi>
Frictionless <frictionless>
Hypothesis <data_synthesis_strategies>
Mypy <mypy_integration>
Pydantic <pydantic_integration>
```

:::{note}
Don't see a library that you want supported? Check out the
[github issues](https://github.com/pandera-dev/pandera/issues) to see if
that library is in the roadmap. If it isn't, open up a
[new issue](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)
to add support for it!
:::
