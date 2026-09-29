"""Placeholder tests for the Phase 0 skeleton: the package imports and the CLI runs."""

from typer.testing import CliRunner

import superposition
from superposition.cli import app


def test_package_has_version() -> None:
    assert isinstance(superposition.__version__, str)


def test_cli_help_runs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.output
