"""The ``superposition`` command-line interface (Typer).

SPEC §5 (CLI):

| Command | What it does |
|---|---|
| ``superposition list`` | list experiments |
| ``superposition run <exp> [--profile quick|full] [--seed N] [--device auto]`` | run one |
| ``superposition figures <exp> [--run latest] [--publish]`` | redraw (and publish) figures |
| ``superposition report`` | regenerate the tables in docs/REPRODUCTION.md |
| ``superposition app`` | launch the Streamlit lab |
"""

from __future__ import annotations

import subprocess
import sys
from typing import Annotated

import typer

from superposition.runs import project_root
from superposition.utils import DEVICE_CHOICES

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
    from superposition.experiments import REGISTRY
    from superposition.runs import latest_run

    for name, exp in REGISTRY.items():
        if names:
            typer.echo(name)
            continue
        run = latest_run(name)
        status = "not run"
        if run is not None and run.acceptance_path.is_file():
            status = f"{run.read_acceptance()['overall']} ({run.run_id})"
        typer.echo(f"{exp.code}  {name:<24} {exp.paper_pages:<12} {status}\n     {exp.title}")


@app.command()
def run(
    experiment: str = typer.Argument(
        ..., help="Experiment name (e.g. intro_pentagon) or code (E00)."
    ),
    profile: str = typer.Option("quick", "--profile", "-p", help="quick or full."),
    seed: int | None = typer.Option(None, "--seed", help="Override the config's seed."),
    device: str = typer.Option("auto", "--device", help=f"One of {', '.join(DEVICE_CHOICES)}."),
    no_figures: bool = typer.Option(False, "--no-figures", help="Skip figure generation."),
) -> None:
    """Run an experiment and write results/<experiment>/<run_id>/."""
    from superposition.experiments.runner import run_experiment

    if profile not in ("quick", "full"):
        raise typer.BadParameter("profile must be 'quick' or 'full'")
    if device not in DEVICE_CHOICES:
        raise typer.BadParameter(f"device must be one of {DEVICE_CHOICES}")
    result = run_experiment(experiment, profile, seed=seed, device=device, figures=not no_figures)
    doc = result.read_acceptance()
    typer.echo(
        f"\n{doc['code']} {doc['experiment']}: {doc['overall']} ({doc['n_passed']}/{doc['n_graded']})"
    )
    typer.echo(f"run directory: {result.path}")


@app.command()
def figures(
    experiment: str = typer.Argument(..., help="Experiment name or code."),
    run_id: str = typer.Option("latest", "--run", help="Run ID, or 'latest'."),
    publish: bool = typer.Option(False, "--publish", help="Copy PNGs to docs/figures/<exp>/."),
) -> None:
    """Redraw an experiment's figures from a saved run."""
    from superposition.experiments.runner import make_figures

    try:
        paths = make_figures(experiment, run_id, publish=publish)
    except FileNotFoundError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    for path in paths:
        typer.echo(str(path))


@app.command()
def report(
    pin: Annotated[
        list[str] | None,
        typer.Option("--pin", help="Cite this experiment's latest complete run (repeatable)."),
    ] = None,
    pin_all: bool = typer.Option(
        False, "--pin-all", help="Cite the latest run of every experiment."
    ),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Allow citing a run made from a dirty working tree."
    ),
) -> None:
    """Regenerate the tables in docs/REPRODUCTION.md from the cited runs."""
    from superposition.experiments import REGISTRY
    from superposition.experiments.runner import get_experiment
    from superposition.report import pin_run, update_reproduction

    names = list(REGISTRY) if pin_all else [get_experiment(p).name for p in pin or []]
    for name in names:
        try:
            citation = pin_run(name, allow_dirty=allow_dirty)
        except (FileNotFoundError, ValueError) as error:
            typer.echo(f"cannot pin {name}: {error}", err=True)
            raise typer.Exit(1) from error
        typer.echo(f"cited {name}: {citation.run_id} ({citation.overall})")
    for note in update_reproduction():
        typer.echo(note)
    typer.echo("updated docs/REPRODUCTION.md")


@app.command("app")
def launch_app(
    port: int = typer.Option(8501, "--port", help="Port to serve the lab on."),
    headless: bool = typer.Option(False, "--headless", help="Do not open a browser tab."),
) -> None:
    """Launch the Streamlit lab (app/Home.py)."""
    root = project_root()
    home = root / "app" / "Home.py"
    if not home.is_file():
        typer.echo(f"app/Home.py not found under {root}; run from a repository checkout.", err=True)
        raise typer.Exit(1)
    command = [
        sys.executable, "-m", "streamlit", "run", str(home),
        "--server.port", str(port), "--server.headless", str(headless).lower(),
    ]  # fmt: skip
    raise typer.Exit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    app()
