"""Command-line interface implementation (requires ``typer``; see ``pandera[cli]``)."""

from __future__ import annotations

import typer

from .generate import generate
from .infer import infer
from .validate import validate

__all__ = ["app", "run"]

app = typer.Typer(
    help="Pandera command-line tools.",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Pandera command-line tools."""


app.command()(validate)
app.command()(infer)
app.command()(generate)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
