"""Intra-batch fusion: roots[p] = first pixel seeing p's surface (grid kept: IRLS matches by pixel index).

Projection-and-depth-gate adpated from ScaRF-SLAM (https://github.com/ori-drs/ScaRF-SLAM,
Zhang et al., https://arxiv.org/abs/2606.00307). We keep the first pixel of each group rather
than averaging, so correspondences stay on the pixel grid for IRLS.
"""

import numpy as np
import torch


def _to_t(a, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def compute_fusion_roots(pts_cam, confs, poses, K, *, depth_rtol=0.03, max_prev_frames=4):
    """Match each frame's valid pixels into up to max_prev_frames earlier frames by projection and depth gate."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, H, W, _ = pts_cam.shape
    N = H * W

    roots = torch.arange(S * N, device=dev, dtype=torch.int64)
    matched = torch.zeros(S * N, dtype=torch.bool, device=dev)

    pts = _to_t(pts_cam.reshape(S, N, 3), dev)
    depth = pts[..., 2].contiguous()  # (S, N)
    valid = _to_t(confs.reshape(S, N), dev) > 0
    Kt = _to_t(K, dev)
    Pt = _to_t(poses, dev, dtype=torch.float64)

    for i in range(1, S):
        for j in range(i - 1, max(-1, i - max_prev_frames - 1), -1):
            cand = valid[i] & ~matched[i * N : (i + 1) * N]
            if not bool(cand.any()):
                break

            # frame i camera -> frame j camera
            T = torch.linalg.inv(Pt[j]) @ Pt[i]
            T = T.to(torch.float32)
            q = pts[i] @ T[:3, :3].T + T[:3, 3]
            z = q[:, 2]

            ok = cand & (z > 1e-6)
            if not bool(ok.any()):
                continue

            fx, fy = Kt[j][0, 0], Kt[j][1, 1]
            cx, cy = Kt[j][0, 2], Kt[j][1, 2]
            u = torch.round(fx * q[:, 0] / z + cx).long()
            v = torch.round(fy * q[:, 1] / z + cy).long()
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not bool(ok.any()):
                continue

            tgt = v.clamp(0, H - 1) * W + u.clamp(0, W - 1)
            d = depth[j][tgt]
            diff = (z - d).abs()
            ok &= valid[j][tgt] & (d > 1e-6) & (diff < depth_rtol * d)
            if not bool(ok.any()):
                continue

            # at most one source pixel per target pixel (best depth agreement), else regions collapse at depth edges
            src = torch.nonzero(ok, as_tuple=False).squeeze(1)
            best = torch.full((N,), float("inf"), device=dev)
            best.scatter_reduce_(0, tgt[src], diff[src], reduce="amin", include_self=True)
            is_best = diff[src] == best[tgt[src]]
            src = src[is_best]
            if src.numel() == 0:
                continue
            # equal diffs can leave two sources on one target: keep the first
            uniq_tgt, first = np.unique(tgt[src].cpu().numpy(), return_index=True)
            src = src[torch.as_tensor(first, device=dev, dtype=torch.int64)]
            tgt_sel = torch.as_tensor(uniq_tgt, device=dev, dtype=torch.int64)

            # frames are visited in increasing i with j < i, so frame j's roots are final
            roots[i * N + src] = roots[j * N + tgt_sel]
            matched[i * N + src] = True

    return roots.cpu().numpy()


def fusion_keep(confs, roots):
    """Bool mask keeping the lowest-index valid (conf > 0) member of every group."""
    valid = np.flatnonzero(confs.reshape(-1) > 0)
    keep = np.zeros(roots.shape[0], dtype=bool)
    if valid.size:
        keep[valid[np.unique(roots[valid], return_index=True)[1]]] = True
    return keep
