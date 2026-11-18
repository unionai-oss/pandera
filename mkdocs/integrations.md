# Ecosystem Integrations

Pandera ships with integrations with other tools in the Python ecosystem, with
the goal of interoperating with libraries that you know and love.

| Integration | Description |
| :---------- | :---------- |
| [FastAPI](./fastapi.html) | Use pandera DataFrameModels in your FastAPI app |
| [Frictionless](./frictionless.html) | Convert frictionless schemas to pandera schemas |
| [Hypothesis](./data_synthesis_strategies.html) | Use the hypothesis library to generate valid data under your schema's constraints. |
| [Mypy](./mypy_integration.html) | Type-lint your pandas and pandera code with mypy for static type safety [experimental 🧪] |
| [Pydantic](./pydantic_integration.html) | Use pandera DataFrameModels when defining  your pydantic BaseModels |
| [Decisions](./decisions.html) | Fill columns by asking a decision model (TypeSafe Jev) typed questions |

!!! note

    Don't see a library that you want supported? Check out the
    [github issues](https://github.com/pandera-dev/pandera/issues) to see if
    that library is in the roadmap. If it isn't, open up a
    [new issue](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)
    to add support for it!
