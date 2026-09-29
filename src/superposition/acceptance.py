"""Acceptance criteria: values, thresholds and PASS / PARTIAL / FAIL statuses.

SPEC §2 (Honest reporting) and §5 (``acceptance.json``: "each criterion with its value,
threshold and status"). Experiments build a list of :class:`Criterion` from their results;
:func:`acceptance_document` turns it into the JSON stored in the run directory, which
``superposition report`` later renders into docs/REPRODUCTION.md.

Statuses: each graded criterion is PASS or FAIL. INFO rows report a number without a
threshold (e.g. "report where antipodal pairs appear") and never affect the overall status.
The overall status is PASS if every graded criterion passes, FAIL if none does, and PARTIAL
otherwise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

PASS, PARTIAL, FAIL, INFO = "PASS", "PARTIAL", "FAIL", "INFO"


@dataclass
class Criterion:
    """One acceptance criterion and its outcome."""

    id: str
    description: str
    paper_page: str
    status: str
    value: Any = None
    threshold: Any = None
    group: str = ""
    notes: str = ""


def to_jsonable(value: Any) -> Any:
    """Convert tensors, arrays, NumPy scalars, sets and tuples to JSON-friendly values."""
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    return value


def check(
    id: str,
    description: str,
    paper_page: str,
    passed: bool,
    value: Any,
    threshold: Any,
    group: str = "",
    notes: str = "",
) -> Criterion:
    """A graded criterion: PASS if ``passed`` else FAIL."""
    return Criterion(
        id=id,
        description=description,
        paper_page=paper_page,
        status=PASS if passed else FAIL,
        value=to_jsonable(value),
        threshold=to_jsonable(threshold),
        group=group,
        notes=notes,
    )


def info(
    id: str, description: str, paper_page: str, value: Any, group: str = "", notes: str = ""
) -> Criterion:
    """A report-only row (status INFO)."""
    return Criterion(
        id=id,
        description=description,
        paper_page=paper_page,
        status=INFO,
        value=to_jsonable(value),
        group=group,
        notes=notes,
    )


def overall_status(criteria: list[Criterion]) -> str:
    """PASS if all graded criteria pass, FAIL if none does, PARTIAL otherwise."""
    graded = [c.status for c in criteria if c.status != INFO]
    if not graded:
        return INFO
    if all(s == PASS for s in graded):
        return PASS
    if all(s == FAIL for s in graded):
        return FAIL
    return PARTIAL


def acceptance_document(
    *,
    experiment: str,
    code: str,
    title: str,
    paper_pages: str,
    run_id: str,
    profile: str,
    criteria: list[Criterion],
) -> dict[str, Any]:
    """The ``acceptance.json`` payload for one run."""
    graded = [c for c in criteria if c.status != INFO]
    return {
        "experiment": experiment,
        "code": code,
        "title": title,
        "paper_pages": paper_pages,
        "run_id": run_id,
        "profile": profile,
        "overall": overall_status(criteria),
        "n_graded": len(graded),
        "n_passed": sum(c.status == PASS for c in graded),
        "criteria": [asdict(c) for c in criteria],
    }
