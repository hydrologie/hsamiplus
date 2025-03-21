"""Console script for hsamiplus."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()  # type: ignore[misc]
def main() -> None:
    """Console script for hsamiplus."""
    console.print(
        "Replace this message by putting your code into hsamiplus.cli.main",
    )
    console.print("See Typer documentation at https://typer.tiangolo.com/")


if __name__ == "__main__":
    app()
