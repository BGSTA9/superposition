"""Unit tests for runs.py, acceptance.py, report.py and the CLI (no training)."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from superposition.acceptance import (
    FAIL,
    INFO,
    PARTIAL,
    PASS,
    acceptance_document,
    check,
    info,
    overall_status,
    to_jsonable,
)
from superposition.cli import app
from superposition.config import ExperimentConfig
from superposition.report import criteria_table, replace_block, status_rows, status_table
from superposition.runs import (
    RESULTS_ENV,
    create_run,
    finish_run,
    latest_run,
    list_runs,
    resolve_run,
)


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(RESULTS_ENV, str(tmp_path / "results"))
    return tmp_path / "results"


def test_run_directory_lifecycle(results: Path) -> None:
    config = ExperimentConfig(experiment="demo", seed=4)
    run = create_run("demo", config, "cpu", seed=4)
    assert run.path.parent == results / "demo"
    assert run.config_path.is_file()
    manifest = run.read_manifest()
    assert manifest["status"] == "running"
    assert manifest["seed"] == 4
    assert {"commit", "dirty"} <= set(manifest["git"])
    assert manifest["packages"]["torch"] == torch.__version__
    assert latest_run("demo") is None  # still running
    run.write_table(pd.DataFrame({"a": [1, 2]}))
    assert run.read_table()["a"].tolist() == [1, 2]
    run.save_checkpoint("model", {"W": torch.ones(2, 3), "kind": "linear"})
    loaded = run.load_checkpoint("model")
    assert torch.equal(loaded["W"], torch.ones(2, 3)) and loaded["kind"] == "linear"
    finish_run(run)
    manifest = run.read_manifest()
    assert manifest["status"] == "complete" and manifest["wall_clock_s"] >= 0
    assert latest_run("demo") == run
    assert resolve_run("demo", "latest") == run
    assert resolve_run("demo", run.run_id) == run
    # Same config within the same second: the run ID gets a suffix instead of colliding.
    second = create_run("demo", config, "cpu", seed=4)
    assert second.path != run.path
    finish_run(second, "failed", error="boom")
    assert list_runs("demo") == [run]
    assert len(list_runs("demo", status=None)) == 2


def test_resolve_run_names_the_command(results: Path) -> None:
    with pytest.raises(FileNotFoundError, match="superposition run nothing"):
        resolve_run("nothing")


def test_overall_status_and_jsonable() -> None:
    passed = check("a", "d", "p. 1", True, 1.0, 2.0)
    failed = check("b", "d", "p. 1", False, torch.tensor([1.0, 2.0]), {"max": 1})
    note = info("c", "d", "p. 1", {3, 1})
    assert (passed.status, failed.status, note.status) == (PASS, FAIL, INFO)
    assert failed.value == [1.0, 2.0]
    assert note.value == [1, 3]
    assert overall_status([passed, note]) == PASS
    assert overall_status([failed, note]) == FAIL
    assert overall_status([passed, failed]) == PARTIAL
    assert overall_status([note]) == INFO
    assert to_jsonable(float("nan")) == "nan"
    doc = acceptance_document(
        experiment="x", code="E99", title="t", paper_pages="p. 1", run_id="r", profile="quick",
        criteria=[passed, failed, note],
    )  # fmt: skip
    assert (doc["overall"], doc["n_passed"], doc["n_graded"]) == (PARTIAL, 1, 2)


def test_report_blocks() -> None:
    text = "intro\n<!-- BEGIN AUTO:E00 -->\nold\n<!-- END AUTO:E00 -->\noutro"
    new, found = replace_block(text, "E00", "new table")
    assert found and "new table" in new and "old" not in new and new.endswith("outro")
    _, found = replace_block(text, "E01", "x")
    assert not found
    table = status_table(status_rows({}))
    assert "not run" in table and "intro_pentagon" in table
    doc = acceptance_document(
        experiment="x", code="E99", title="t", paper_pages="p. 1", run_id="r1", profile="quick",
        criteria=[check("a", "desc | with pipe", "p. 1", True, [1, 2], None)],
    )  # fmt: skip
    rendered = criteria_table(doc, {"git": {"commit": "abc", "dirty": False}, "seed": 0})
    assert "desc \\| with pipe" in rendered and "`r1`" in rendered and "PASS" in rendered


def test_cli_list_figures_and_app(results: Path) -> None:
    runner = CliRunner()
    names = runner.invoke(app, ["list", "--names"])
    assert names.exit_code == 0
    assert names.output.split() == ["intro_pentagon", "basic_results"]
    missing = runner.invoke(app, ["figures", "intro_pentagon"])
    assert missing.exit_code == 1
    bad_profile = runner.invoke(app, ["run", "intro_pentagon", "--profile", "medium"])
    assert bad_profile.exit_code != 0


def fake_complete_run(overall: str = "PASS", dirty: bool = False):
    """A complete intro_pentagon run with a one-criterion acceptance record."""
    from superposition.runs import create_run, finish_run

    run = create_run("intro_pentagon", ExperimentConfig(experiment="intro_pentagon"), "cpu", seed=0)
    run.update_manifest(git={"commit": "abc123", "dirty": dirty})
    criterion = check("c", "a criterion", "p. 2", overall == "PASS", 1.0, 2.0)
    document = acceptance_document(
        experiment="intro_pentagon", code="E00", title="t", paper_pages="p. 2",
        run_id=run.run_id, profile="quick", criteria=[criterion],
    )  # fmt: skip
    run.write_acceptance(document)
    finish_run(run)
    return run


def test_pinning_and_regenerating_reproduction(results: Path, tmp_path: Path) -> None:
    from superposition.report import load_citations, pin_run, update_reproduction

    cited_file = tmp_path / "cited_runs.yaml"
    run = fake_complete_run()
    citation = pin_run("intro_pentagon", path=cited_file)
    assert (citation.run_id, citation.overall, citation.commit) == (run.run_id, "PASS", "abc123")
    assert load_citations(cited_file)["intro_pentagon"] == citation

    # A newer run does not change what the docs cite.
    newer = fake_complete_run(overall="FAIL")
    citations = load_citations(cited_file)
    rows = {r.name: r for r in status_rows(citations)}
    assert rows["intro_pentagon"].source == "cited"
    assert rows["intro_pentagon"].citation.run_id == run.run_id  # type: ignore[union-attr]
    assert rows["basic_results"].source == "not run"

    doc = tmp_path / "REPRODUCTION.md"
    doc.write_text(
        "<!-- BEGIN AUTO:STATUS -->\nold\n<!-- END AUTO:STATUS -->\n"
        "<!-- BEGIN AUTO:E00 -->\nold\n<!-- END AUTO:E00 -->\n"
    )
    update_reproduction(doc, citations)
    text = doc.read_text()
    assert f"`{run.run_id}`" in text and newer.run_id not in text
    assert "**PASS**" in text and "a criterion" in text

    # Cited run missing locally (e.g. a fresh clone): the status row survives, the block stays.
    import shutil

    shutil.rmtree(run.path)
    doc.write_text(doc.read_text().replace("a criterion", "hand-kept"))
    notes = update_reproduction(doc, citations)
    assert "hand-kept" in doc.read_text()
    assert any("not in" in note for note in notes)
    assert f"`{run.run_id}`" in doc.read_text()


def test_pinning_refuses_dirty_runs(results: Path, tmp_path: Path) -> None:
    from superposition.report import pin_run

    fake_complete_run(dirty=True)
    with pytest.raises(ValueError, match="dirty"):
        pin_run("intro_pentagon", path=tmp_path / "c.yaml")
    assert pin_run("intro_pentagon", allow_dirty=True, path=tmp_path / "c.yaml").dirty is True
    with pytest.raises(FileNotFoundError):
        pin_run("basic_results", path=tmp_path / "c.yaml")
