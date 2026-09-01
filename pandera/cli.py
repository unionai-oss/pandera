"""Command-line interface for Pandera."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``pandera`` console script and ``python -m pandera``."""
    try:
        import typer  # noqa: F401
    except ImportError as exc:
        sys.stderr.write(
            "The pandera CLI requires typer. Install with:\n"
            "  pip install 'pandera[cli]'\n"
        )
        raise SystemExit(1) from exc
    from pandera._cli import run

    run()


if __name__ == "__main__":
    main()
