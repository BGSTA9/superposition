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
def report() -> None:
    """Regenerate the tables in docs/REPRODUCTION.md from the latest runs."""
    from superposition.report import update_reproduction

    typer.echo(f"updated {update_reproduction()}")


@app.command("app")
def launch_app() -> None:
    """Launch the Streamlit lab (app/Home.py)."""
    home = project_root() / "app" / "Home.py"
    if not home.is_file():
        typer.echo(
            "The lab is built in Phase 2 (see PLAN.md); app/Home.py does not exist yet.", err=True
        )
        raise typer.Exit(1)
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(home)]))
