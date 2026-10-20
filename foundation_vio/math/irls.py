"""Robust Sim(3) between point maps on one pixel grid: IRLS, Huber, confidence weights, one fused numba pass.

Adapted from VGGT-Long (https://github.com/DengKaiCQ/VGGT-Long, Deng et al., https://arxiv.org/abs/2507.16443):
confidence-weighted Huber IRLS with a closed-form Umeyama step each iteration.
"""

import numpy as np
from numba import get_num_threads, njit, prange

# partial-sum layout per chunk: sum w, sum w x (3), sum w y (3), sum w|x|^2, sum w|y|^2, sum w x y^T (9), Huber loss
_W, _SX, _SY, _SXX, _SYY, _SXY, _LOSS, _NPART = 0, 1, 4, 7, 8, 9, 18, 19


@njit(cache=True, fastmath=True, parallel=True)
def _fused_pass(src, tgt, conf1, conf2, thresh, s, R, t, delta, use_transform, partials):
    n = src.shape[0]
    n_chunks = partials.shape[0]
    chunk = (n + n_chunks - 1) // n_chunks
    r00, r01, r02 = s * R[0, 0], s * R[0, 1], s * R[0, 2]
    r10, r11, r12 = s * R[1, 0], s * R[1, 1], s * R[1, 2]
    r20, r21, r22 = s * R[2, 0], s * R[2, 1], s * R[2, 2]
    t0, t1, t2 = t[0], t[1], t[2]

    for c in prange(n_chunks):
        lo = c * chunk
        hi = min(n, lo + chunk)
        acc = np.zeros(_NPART)
        for i in range(lo, hi):
            c1 = conf1[i]
            c2 = conf2[i]
            if c1 <= thresh or c2 <= thresh:
                continue
            w0 = np.sqrt(c1 * c2)
            x0, x1, x2 = src[i, 0], src[i, 1], src[i, 2]
            y0, y1, y2 = tgt[i, 0], tgt[i, 1], tgt[i, 2]
            w = w0
            if use_transform:
                d0 = y0 - (r00 * x0 + r01 * x1 + r02 * x2 + t0)
                d1 = y1 - (r10 * x0 + r11 * x1 + r12 * x2 + t1)
                d2 = y2 - (r20 * x0 + r21 * x1 + r22 * x2 + t2)
                r = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                if r > delta:
                    w = w0 * (delta / r)
                    acc[_LOSS] += w0 * delta * (r - 0.5 * delta)
                else:
                    acc[_LOSS] += w0 * 0.5 * r * r
            acc[_W] += w
            acc[_SX] += w * x0
            acc[_SX + 1] += w * x1
            acc[_SX + 2] += w * x2
            acc[_SY] += w * y0
            acc[_SY + 1] += w * y1
            acc[_SY + 2] += w * y2
            acc[_SXX] += w * (x0 * x0 + x1 * x1 + x2 * x2)
            acc[_SYY] += w * (y0 * y0 + y1 * y1 + y2 * y2)
            acc[_SXY + 0] += w * x0 * y0
            acc[_SXY + 1] += w * x0 * y1
            acc[_SXY + 2] += w * x0 * y2
            acc[_SXY + 3] += w * x1 * y0
            acc[_SXY + 4] += w * x1 * y1
            acc[_SXY + 5] += w * x1 * y2
            acc[_SXY + 6] += w * x2 * y0
            acc[_SXY + 7] += w * x2 * y1
            acc[_SXY + 8] += w * x2 * y2
        partials[c, :] = acc


@njit(cache=True)
def _sim3_from_moments(m):
    """Closed-form weighted Umeyama from raw moments: (s, R, t, ok)."""
    W = m[_W]
    if W < 1e-6:
        return -1.0, np.eye(3), np.zeros(3), False
    mu_x = m[_SX : _SX + 3] / W
    mu_y = m[_SY : _SY + 3] / W
    var_x = m[_SXX] / W - mu_x @ mu_x
    var_y = m[_SYY] / W - mu_y @ mu_y
    s = np.sqrt(var_y / var_x)
    H = s * (m[_SXY : _SXY + 9].reshape(3, 3) / W - np.outer(mu_x, mu_y))
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_y - s * (R @ mu_x)
    return s, R, t, True


def robust_sim3(src, tgt, conf1, conf2, thresh, *, delta=0.1, max_iters=5, tol=1e-9):
    """Sim(3) (s, R, t) mapping src onto tgt; conf1 belongs to src, conf2 to tgt."""
    src = np.ascontiguousarray(src.reshape(-1, 3), dtype=np.float32)
    tgt = np.ascontiguousarray(tgt.reshape(-1, 3), dtype=np.float32)
    conf1 = np.ascontiguousarray(conf1.reshape(-1), dtype=np.float32)
    conf2 = np.ascontiguousarray(conf2.reshape(-1), dtype=np.float32)
    partials = np.empty((4 * get_num_threads(), _NPART))

    _fused_pass(src, tgt, conf1, conf2, thresh, 1.0, np.eye(3), np.zeros(3), delta, False, partials)  # noqa: FBT003
    s, R, t, ok = _sim3_from_moments(partials.sum(axis=0))
    if not ok:
        raise ValueError("No matching point pairs were found!")

    prev_error = np.inf
    cos_tol = np.cos(np.radians(0.1))
    for _ in range(max_iters):
        _fused_pass(src, tgt, conf1, conf2, thresh, s, R, t, delta, True, partials)  # noqa: FBT003
        m = partials.sum(axis=0)
        s_new, R_new, t_new, ok = _sim3_from_moments(m)
        if not ok:
            break
        current_error = m[_LOSS]
        param_change = abs(s_new - s) + np.linalg.norm(t_new - t)
        cos_rot = (np.trace(R_new @ R.T) - 1.0) / 2.0
        if (param_change < tol and cos_rot > cos_tol) or abs(prev_error - current_error) < tol * prev_error:
            break
        s, R, t = s_new, R_new, t_new
        prev_error = current_error
    return s, R, t


def estimate_scale(prev_pts, prev_confs, curr_pts, curr_confs):
    """Scale s with s * curr ~ prev, from two point maps of the same image."""
    # confidence cut at 10 % of the smaller median, taken on a 1/16 subsample (within ~0.4 % of the full median)
    thresh = min(np.median(prev_confs.ravel()[::16]), np.median(curr_confs.ravel()[::16])) * 0.1
    return robust_sim3(curr_pts, prev_pts, curr_confs, prev_confs, thresh)[0]


def warmup():
    """Compile the numba kernels before the first batch."""
    src = np.random.default_rng(0).standard_normal((1024, 3)).astype(np.float32)
    conf = np.ones(1024, dtype=np.float32)
    robust_sim3(src, src, conf, conf, 0.5)
