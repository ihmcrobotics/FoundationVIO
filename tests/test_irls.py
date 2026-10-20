"""The robust Sim(3) recovers a planted scale, rotation and translation despite outliers."""

import numpy as np

from foundation_vio.math.irls import estimate_scale, robust_sim3


def _seam(n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.5, 8.0, n)
    src = np.stack([rng.uniform(-0.7, 0.7, n) * depth, rng.uniform(-0.5, 0.5, n) * depth, depth], axis=1)
    ang = np.radians(3.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    tgt = 1.37 * src @ R.T + [0.1, -0.05, 0.2] + rng.normal(0, 0.01, (n, 3))
    outliers = rng.random(n) < 0.05
    tgt[outliers] += rng.normal(0, 1.0, (outliers.sum(), 3))
    conf1, conf2 = rng.lognormal(0, 0.7, n), rng.lognormal(0, 0.7, n)
    return src.astype(np.float32), tgt.astype(np.float32), conf1.astype(np.float32), conf2.astype(np.float32), R


def test_recovers_planted_sim3():
    src, tgt, conf1, conf2, R_true = _seam()
    s, R, t = robust_sim3(src, tgt, conf1, conf2, 0.1 * min(np.median(conf1), np.median(conf2)))
    assert abs(s - 1.37) < 0.01
    assert np.degrees(np.arccos(np.clip((np.trace(R @ R_true.T) - 1) / 2, -1, 1))) < 0.05
    assert np.linalg.norm(t - [0.1, -0.05, 0.2]) < 0.02


def test_estimate_scale_on_grids():
    src, tgt, conf1, conf2, _ = _seam(n=2 * 100 * 60)
    grid = lambda a: a.reshape(2, 100, 60, -1) if a.ndim == 2 else a.reshape(2, 100, 60)  # noqa: E731
    assert abs(estimate_scale(grid(tgt), grid(conf2), grid(src), grid(conf1)) - 1.37) < 0.02
