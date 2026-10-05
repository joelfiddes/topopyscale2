"""The ``tps2`` Typer application, shared by the command modules."""

import sys
from typing import Optional

import typer
from rich.console import Console

import topopyscale2


def _utf8_streams() -> None:
    """Write UTF-8 to stdout/stderr whatever the platform's default.

    On Windows, output that goes to a pipe or a file (a log, `tps2 ui`'s run log) is encoded
    with the locale code page, e.g. cp1252, which cannot encode the box drawing and arrows in
    the help and progress output: `tps2 init --help` crashed with UnicodeEncodeError. Must run
    before the Console below is created.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):   # a stream that cannot be reconfigured
                pass


_utf8_streams()

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
