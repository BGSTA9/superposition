"""Builds the tables in docs/REPRODUCTION.md from cited runs' acceptance results.

SPEC §10: "``superposition report`` regenerates the tables; the narrative is written by hand
from actual results." Only text between ``<!-- BEGIN AUTO:<key> -->`` and
``<!-- END AUTO:<key> -->`` markers is replaced: ``STATUS`` holds the summary table, and one
block per experiment code (``E00``, ``E01``, …) holds that experiment's criteria. A missing
experiment block is appended under a new heading, ready for the hand-written narrative.

**Cited runs.** The hand-written narrative quotes numbers from specific runs, so the tables
must come from those same runs, not from whatever ran last (a notebook, another device, a
dirty tree). ``docs/cited_runs.yaml`` pins one run per experiment and stores its status
summary, so the status table can be rebuilt even on a fresh clone without ``results/``. Pin
with ``superposition report --pin <experiment>`` (latest clean complete run) after writing
the narrative for it. Experiments without a pin fall back to their latest local run, marked
as uncited.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from superposition.experiments import REGISTRY
from superposition.runs import RunDir, latest_run, project_root, results_root

STATUS_KEY = "STATUS"
CITATIONS_RELATIVE = Path("docs") / "cited_runs.yaml"


def _marker(key: str) -> tuple[str, str]:
    return f"<!-- BEGIN AUTO:{key} -->", f"<!-- END AUTO:{key} -->"


def _cell(value: Any, limit: int = 140) -> str:
    """Compact, table-safe rendering of a criterion value."""
    if value is None:
        return "—"
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    )
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- citations -------------------------------------------------------------------------------


@dataclass
class Citation:
    """The run REPRODUCTION.md cites for one experiment, with its status summary."""

    run_id: str
    profile: str
    device: str
    seed: int
    commit: str | None
    dirty: bool | None
    overall: str
    n_passed: int
    n_graded: int
    wall_clock_s: float | None


def citations_path() -> Path:
    return project_root() / CITATIONS_RELATIVE


def load_citations(path: Path | None = None) -> dict[str, Citation]:
    """Read ``docs/cited_runs.yaml`` (empty if it does not exist)."""
    path = path or citations_path()
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {name: Citation(**entry) for name, entry in raw.items()}


def save_citations(citations: dict[str, Citation], path: Path | None = None) -> Path:
    path = path or citations_path()
    header = (
        "# Runs cited by docs/REPRODUCTION.md, one per experiment (written by\n"
        "# `superposition report --pin <experiment>`; do not edit by hand).\n"
    )
    body = yaml.safe_dump({k: asdict(v) for k, v in sorted(citations.items())}, sort_keys=False)
    path.write_text(header + body)
    return path


def citation_for(run: RunDir) -> Citation:
    """Summarize a complete run as a citation."""
    manifest = run.read_manifest()
    doc = run.read_acceptance()
    git = manifest.get("git", {})
    return Citation(
        run_id=run.run_id,
        profile=doc["profile"],
        device=str(manifest.get("device")),
        seed=int(manifest.get("seed", 0)),
        commit=git.get("commit"),
        dirty=git.get("dirty"),
        overall=doc["overall"],
        n_passed=int(doc["n_passed"]),
        n_graded=int(doc["n_graded"]),
        wall_clock_s=manifest.get("wall_clock_s"),
    )


def pin_run(
    experiment: str,
    run_id: str = "latest",
    allow_dirty: bool = False,
    path: Path | None = None,
) -> Citation:
    """Cite a run of ``experiment`` (default: its latest complete run).

    Raises:
        FileNotFoundError: if the run does not exist or is not complete.
        ValueError: if the run was made from a dirty tree and ``allow_dirty`` is False
            (SPEC §2: numbers in the docs must trace to a committed state).
    """
    if run_id == "latest":
        run = latest_run(experiment)
        if run is None:
            raise FileNotFoundError(
                f"no complete run of {experiment!r}; create one with "
                f"`uv run superposition run {experiment}`"
            )
    else:
        run = RunDir(results_root() / experiment / run_id)
        if run.status != "complete":
            raise FileNotFoundError(f"run {run_id!r} of {experiment!r} is missing or incomplete")
    citation = citation_for(run)
    if citation.dirty and not allow_dirty:
        raise ValueError(
            f"run {run.run_id} was made from a dirty working tree; commit first and re-run, "
            "or pass --allow-dirty"
        )
    citations = load_citations(path)
    citations[experiment] = citation
    save_citations(citations, path)
    return citation


# --- tables ----------------------------------------------------------------------------------


@dataclass
class StatusRow:
    """What the status table shows for one experiment."""

    code: str
    name: str
    paper_pages: str
    source: str  # "cited", "latest (uncited)" or "not run"
    citation: Citation | None


def status_rows(citations: dict[str, Citation]) -> list[StatusRow]:
    """Cited run per experiment, else its latest local run, else "not run"."""
    rows = []
    for name, exp in REGISTRY.items():
        if name in citations:
            rows.append(StatusRow(exp.code, name, exp.paper_pages, "cited", citations[name]))
            continue
        run = latest_run(name)
        if run is not None and run.acceptance_path.is_file():
            rows.append(
                StatusRow(exp.code, name, exp.paper_pages, "latest (uncited)", citation_for(run))
            )
        else:
            rows.append(StatusRow(exp.code, name, exp.paper_pages, "not run", None))
    return rows


def status_table(rows: list[StatusRow]) -> str:
    """The summary table in docs/REPRODUCTION.md."""
    lines = [
        "| Code | Experiment | Paper | Status | Graded criteria passed | Run ID | Profile · device |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        c = row.citation
        if c is None:
            lines.append(
                f"| {row.code} | `{row.name}` | {row.paper_pages} | not run | — | — | "
                f"`superposition run {row.name}` |"
            )
            continue
        suffix = "" if row.source == "cited" else " (uncited)"
        lines.append(
            f"| {row.code} | `{row.name}` | {row.paper_pages} | **{c.overall}**{suffix} | "
            f"{c.n_passed}/{c.n_graded} | `{c.run_id}` | {c.profile} · {c.device} |"
        )
    return "\n".join(lines)


def criteria_table(doc: dict[str, Any], manifest: dict[str, Any] | None = None) -> str:
    """One experiment's criteria as a Markdown table, with provenance."""
    lines = []
    if manifest is not None:
        git = manifest.get("git", {})
        commit = (git.get("commit") or "unknown")[:10]
        dirty = " (dirty tree)" if git.get("dirty") else ""
        lines.append(
            f"Run `{doc['run_id']}` · profile `{doc['profile']}` · seed {manifest.get('seed')} · "
            f"device `{manifest.get('device')}` · commit `{commit}`{dirty} · "
            f"wall-clock {manifest.get('wall_clock_s', '?')} s · overall **{doc['overall']}** "
            f"({doc['n_passed']}/{doc['n_graded']})"
        )
        lines.append("")
    lines += [
        "| Group | Criterion | Paper | Status | Value | Threshold |",
        "|---|---|---|---|---|---|",
    ]
    for c in doc["criteria"]:
        lines.append(
            f"| {c.get('group') or '—'} | {_cell(c['description'], 120)} | {c['paper_page']} | "
            f"{c['status']} | {_cell(c.get('value'))} | {_cell(c.get('threshold'), 80)} |"
        )
    return "\n".join(lines)


def replace_block(text: str, key: str, body: str) -> tuple[str, bool]:
    """Replace the body of block ``key``; return (new_text, found)."""
    begin, end = _marker(key)
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda _: f"{begin}\n{body}\n{end}", text), True


def update_reproduction(
    path: Path | None = None, citations: dict[str, Citation] | None = None
) -> list[str]:
    """Regenerate every AUTO block in docs/REPRODUCTION.md; return notes for the user.

    The status table always regenerates (citations carry their own summary). An experiment's
    criteria block regenerates from its cited run if that run exists locally, else from its
    latest local run if it has no citation; otherwise the block is left untouched.
    """
    path = path or project_root() / "docs" / "REPRODUCTION.md"
    citations = load_citations() if citations is None else citations
    notes: list[str] = []
    text = path.read_text()
    rows = status_rows(citations)
    text, found = replace_block(text, STATUS_KEY, status_table(rows))
    if not found:
        begin, end = _marker(STATUS_KEY)
        text += f"\n## Status\n\n{begin}\n{status_table(rows)}\n{end}\n"
    for row in rows:
        exp = REGISTRY[row.name]
        if row.citation is None:
            notes.append(f"{exp.code} {row.name}: not run; its block is unchanged")
            continue
        run = RunDir(results_root() / row.name / row.citation.run_id)
        if not run.acceptance_path.is_file():
            notes.append(
                f"{exp.code} {row.name}: cited run {row.citation.run_id} is not in "
                f"{results_root()}; its criteria block is unchanged"
            )
            continue
        if row.source != "cited":
            notes.append(f"{exp.code} {row.name}: no citation; used the latest local run")
        body = criteria_table(run.read_acceptance(), run.read_manifest())
        text, found = replace_block(text, exp.code, body)
        if not found:
            begin, end = _marker(exp.code)
            text += f"\n## {exp.code} {row.name} ({exp.paper_pages})\n\n{begin}\n{body}\n{end}\n"
    path.write_text(text)
    return notes
