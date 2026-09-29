"""E05 analysis on synthetic trajectories and hand-built geometries (Paper: pp. 26–27)."""

import math

import numpy as np
import torch

from superposition.experiments.learning_dynamics import (
    COLLAPSED,
    OCTAHEDRON,
    PRISM,
    SMALL,
    TRANSITION,
    JumpThresholds,
    StageThresholds,
    canonical_frame_3d,
    find_jumps,
    level_census,
    set_index,
    stage_codes,
    trajectory_type,
)

SETS = set_index(2, 3)


def synthetic_run(jump_at: int, drop: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two features over 20,000 steps (snapshots every 100): feature 0 sits at 1/2 throughout;
    feature 1 sits at 0.05 and jumps to 0.5 over 1,000 steps starting at ``jump_at``. The loss
    drifts down slowly and falls by ``drop`` during the jump."""
    steps = np.arange(0, 20_001, 100)
    d1 = np.clip(0.05 + 0.45 * (steps - jump_at) / 1_000, 0.05, 0.5)
    dims = np.stack([np.full_like(d1, 0.5), d1], axis=1)
    loss = 30.0 - 1e-6 * steps - drop * np.clip((steps - jump_at) / 1_000, 0, 1)
    return steps, dims, loss


def test_jump_with_loss_drop_is_detected() -> None:
    steps, dims, loss = synthetic_run(jump_at=12_000, drop=0.05)
    jumps = find_jumps(steps, dims, loss, JumpThresholds())
    assert [j.feature for j in jumps] == [1]
    jump = jumps[0]
    assert jump.coincides_with_drop
    # the reported window is the first 2,000-step window over which Dᵢ changes by > 0.15
    assert 12_000 - 2_000 <= jump.start_step <= 12_000
    assert jump.end_step - jump.start_step == 2_000
    assert jump.d_end - jump.d_start > 0.15
    assert 0.04 < jump.loss_drop < 0.06


def test_jump_without_loss_drop_does_not_coincide() -> None:
    steps, dims, loss = synthetic_run(jump_at=12_000, drop=0.0)
    jumps = find_jumps(steps, dims, loss, JumpThresholds())
    assert len(jumps) == 1 and not jumps[0].coincides_with_drop


def test_initial_organization_is_not_a_jump() -> None:
    """A change before the feature has held a level for the stable window is not counted."""
    steps, dims, loss = synthetic_run(jump_at=3_000, drop=0.05)
    assert find_jumps(steps, dims, loss, JumpThresholds()) == []


def test_level_census() -> None:
    census = level_census(np.array([0.5, 0.49, 0.0, 0.01, 0.3]), tol=0.05)
    assert census == {"at_1/2": 2, "at_0": 2, "other": 1}


def octahedron() -> torch.Tensor:
    eye = torch.eye(3)
    return torch.cat([eye, -eye], dim=1)  # set A = e₁, e₂, e₃; set B = their antipodes


def triangle(z: float, phase: float) -> torch.Tensor:
    """Three unit vectors at height z, 120° apart in azimuth, starting at ``phase``."""
    r = math.sqrt(1 - z**2)
    angles = torch.tensor([phase + k * 2 * math.pi / 3 for k in range(3)])
    return torch.stack([r * torch.cos(angles), r * torch.sin(angles), torch.full((3,), z)])


def test_stage_codes_on_ideal_geometries() -> None:
    thr = StageThresholds()
    z = 1 / math.sqrt(3)
    geometries = {
        "octahedron": octahedron(),
        # the same octahedron seen as two triangles rotated 60° apart (antiprism)
        "antiprism": torch.cat([triangle(z, 0.0), triangle(-z, math.pi)], dim=1),
        # aligned triangles on opposite sides: cross-set partners at cos −1/3, not antipodal
        "prism": torch.cat([triangle(z, 0.0), triangle(-z, 0.0)], dim=1),
        # each set bunched along one direction, the two directions orthogonal
        "collapsed": torch.cat(
            [
                torch.tensor([[1.0], [0.0], [0.0]]).repeat(1, 3),
                torch.tensor([[0.0], [1.0], [0.0]]).repeat(1, 3),
            ],
            dim=1,
        ),
        "small": 0.05 * octahedron(),
    }
    codes = stage_codes(torch.stack(list(geometries.values())), SETS, thr)
    assert dict(zip(geometries, codes.tolist(), strict=True)) == {
        "octahedron": OCTAHEDRON,
        "antiprism": OCTAHEDRON,
        "prism": PRISM,
        "collapsed": COLLAPSED,
        "small": SMALL,
    }


def test_antipodal_pairs_within_a_set_are_not_the_octahedron() -> None:
    """The paper's octahedron pairs features from *different* sets (p. 27)."""
    wrong = octahedron()[:, [0, 3, 1, 4, 2, 5]]  # set A = {e₁, −e₁, e₂}
    assert stage_codes(wrong[None], SETS, StageThresholds())[0] != OCTAHEDRON


def test_unclassified_geometry_is_a_transition() -> None:
    W = torch.cat([triangle(0.3, 0.0), triangle(0.2, 1.0)], dim=1)  # sets not opposed
    assert stage_codes(W[None], SETS, StageThresholds())[0] == TRANSITION


def test_stage_codes_are_rotation_invariant() -> None:
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    W = torch.stack([octahedron(), torch.cat([triangle(0.5, 0.0), triangle(-0.5, 0.0)], dim=1)])
    thr = StageThresholds()
    assert (stage_codes(W, SETS, thr) == stage_codes(q @ W, SETS, thr)).all()


def test_trajectory_type_drops_short_stages_and_merges() -> None:
    codes = np.array(
        [SMALL] * 2
        + [COLLAPSED] * 30
        + [TRANSITION] * 3
        + [COLLAPSED] * 10
        + [PRISM] * 40
        + [OCTAHEDRON] * 5
    )
    # snapshots every 50 steps; stages under 1,000 steps (20 snapshots) are left out,
    # except the final one
    assert trajectory_type(codes, 50, 1_000) == [COLLAPSED, PRISM, OCTAHEDRON]


def test_canonical_frame_is_a_rotation_putting_set_a_on_top() -> None:
    g = torch.Generator().manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    W = (q @ octahedron()).numpy()
    R = canonical_frame_3d(W, SETS)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    centroid = (R @ W)[:, :3].mean(axis=1)
    assert np.allclose(centroid / np.linalg.norm(centroid), [0, 0, 1], atol=1e-6)
