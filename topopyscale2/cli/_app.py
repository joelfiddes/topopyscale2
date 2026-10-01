"""The ``tps2`` Typer application, shared by the command modules."""

from typing import Optional

import typer
from rich.console import Console

import topopyscale2

app = typer.Typer(
    name="tps2",
    help="TopoPyScale 2.0 -- Topographic downscaling of meteorological forcing",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"TopoPyScale {topopyscale2.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
):
    """TopoPyScale 2.0 -- Topographic downscaling of meteorological forcing."""
    pass
