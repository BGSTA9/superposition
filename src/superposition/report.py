"""Builds the tables in docs/REPRODUCTION.md from the latest runs' acceptance results.

SPEC §10: "``superposition report`` regenerates the tables; the narrative is written by hand
from actual results." Only text between ``<!-- BEGIN AUTO:<key> -->`` and
``<!-- END AUTO:<key> -->`` markers is replaced: ``STATUS`` holds the summary table, and one
block per experiment code (``E00``, ``E01``, …) holds that experiment's criteria. A missing
experiment block is appended under a new heading, ready for the hand-written narrative.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from superposition.experiments import REGISTRY
from superposition.runs import latest_run, project_root

STATUS_KEY = "STATUS"


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


def status_table(documents: dict[str, dict[str, Any] | None]) -> str:
    """The summary table: one row per registered experiment."""
    lines = [
        "| Code | Experiment | Paper | Status | Graded criteria passed | Run ID | Profile |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, exp in REGISTRY.items():
        doc = documents.get(name)
        if doc is None:
            lines.append(
                f"| {exp.code} | `{name}` | {exp.paper_pages} | not run | — | — | "
                f"`superposition run {name}` |"
            )
            continue
        lines.append(
            f"| {exp.code} | `{name}` | {exp.paper_pages} | **{doc['overall']}** | "
            f"{doc['n_passed']}/{doc['n_graded']} | `{doc['run_id']}` | {doc['profile']} |"
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


def collect_documents() -> dict[str, tuple[dict[str, Any], dict[str, Any]] | None]:
    """Latest complete run's (acceptance, manifest) per registered experiment."""
    found: dict[str, tuple[dict[str, Any], dict[str, Any]] | None] = {}
    for name in REGISTRY:
        run = latest_run(name)
        if run is None or not run.acceptance_path.is_file():
            found[name] = None
        else:
            found[name] = (run.read_acceptance(), run.read_manifest())
    return found


def update_reproduction(path: Path | None = None) -> Path:
    """Regenerate every AUTO block in docs/REPRODUCTION.md (creating missing ones)."""
    path = path or project_root() / "docs" / "REPRODUCTION.md"
    text = path.read_text()
    found = collect_documents()
    text, ok = replace_block(
        text, STATUS_KEY, status_table({k: v[0] if v else None for k, v in found.items()})
    )
    if not ok:
        begin, end = _marker(STATUS_KEY)
        text += f"\n## Status\n\n{begin}\n{end}\n"
        text, _ = replace_block(
            text, STATUS_KEY, status_table({k: v[0] if v else None for k, v in found.items()})
        )
    for name, exp in REGISTRY.items():
        entry = found[name]
        if entry is None:
            continue
        body = criteria_table(*entry)
        text, ok = replace_block(text, exp.code, body)
        if not ok:
            begin, end = _marker(exp.code)
            text += f"\n## {exp.code} {name} ({exp.paper_pages})\n\n{begin}\n{body}\n{end}\n"
    path.write_text(text)
    return path
