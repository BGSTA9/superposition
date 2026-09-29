"""The `superposition` command-line interface (Typer).

SPEC §5 (CLI). Phase 0 provides the command skeleton; the commands are implemented in
Phase 1 (list, run, figures) and Phase 2 (report, app).
"""

import typer

app = typer.Typer(
    help="Replication of Toy Models of Superposition (Elhage et al., 2022).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Replicate, explore and interpret toy models of superposition."""


@app.command("list")
def list_experiments(
    names: bool = typer.Option(False, "--names", help="Print bare experiment names only."),
) -> None:
    """List registered experiments."""
    del names  # no experiments are registered until Phase 1
    typer.echo("No experiments registered yet (Phase 0 skeleton).")
