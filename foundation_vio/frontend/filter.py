# The depth and normal edge filters are adapted from MapAnything (https://github.com/facebookresearch/map-anything),
# Copyright (c) Meta Platforms, Inc. and affiliates, licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). Modified for FoundationVIO; the rest of this file is GPL-3.0.
"""Depth/normal edge (MapAnything geometry utils, Apache-2.0) and range filters; bit-identical plane kernels."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter


def _max_pool_2d(x: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Max over a kernel_size x kernel_size neighborhood, edge-padded."""
    return maximum_filter(x, size=kernel_size, mode="nearest")


def depth_edge(
    depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3, mask: np.ndarray | None = None
) -> np.ndarray:
    """Mark (H, W) pixels whose neighborhood spans a relative depth jump larger than rtol."""
    if mask is None:
        mask = np.isfinite(depth)
    hi = _max_pool_2d(np.where(mask, depth, -np.inf), kernel_size)
    lo = -_max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (hi - lo) / depth > rtol


def _points_to_normal_planes(
    points: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normals (nx, ny, nz, valid) of an (H, W, 3) point grid from cross products of the 4 neighbour pairs."""
    height, width = points.shape[:2]
    dtype = points.dtype

    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask

    px = np.zeros((height + 2, width + 2), dtype=dtype)
    py = np.zeros((height + 2, width + 2), dtype=dtype)
    pz = np.zeros((height + 2, width + 2), dtype=dtype)
    px[1:-1, 1:-1] = points[..., 0]
    py[1:-1, 1:-1] = points[..., 1]
    pz[1:-1, 1:-1] = points[..., 2]

    ctr = (slice(1, -1), slice(1, -1))
    cx, cy, cz = px[ctr], py[ctr], pz[ctr]
    up = (px[:-2, 1:-1] - cx, py[:-2, 1:-1] - cy, pz[:-2, 1:-1] - cz)
    left = (px[1:-1, :-2] - cx, py[1:-1, :-2] - cy, pz[1:-1, :-2] - cz)
    down = (px[2:, 1:-1] - cx, py[2:, 1:-1] - cy, pz[2:, 1:-1] - cz)
    right = (px[1:-1, 2:] - cx, py[1:-1, 2:] - cy, pz[1:-1, 2:] - cz)

    m_up = mask_pad[:-2, 1:-1]
    m_left = mask_pad[1:-1, :-2]
    m_down = mask_pad[2:, 1:-1]
    m_right = mask_pad[1:-1, 2:]
    m_ctr = mask_pad[ctr]

    valids = (
        m_up & m_left & m_ctr,
        m_left & m_down & m_ctr,
        m_down & m_right & m_ctr,
        m_right & m_up & m_ctr,
    )
    pairs = ((up, left), (left, down), (down, right), (right, up))

    ax = ay = az = None
    for (u, v), val in zip(pairs, valids, strict=True):
        gx = u[1] * v[2] - u[2] * v[1]
        gy = u[2] * v[0] - u[0] * v[2]
        gz = u[0] * v[1] - u[1] * v[0]
        d = np.sqrt(gx * gx + gy * gy + gz * gz)
        d += 1e-12
        gx /= d
        gy /= d
        gz /= d
        gx *= val  # bool multiply, as in the original (keeps NaNs from bad inputs)
        gy *= val
        gz *= val
        if ax is None:
            ax, ay, az = gx, gy, gz
        else:
            ax += gx
            ay += gy
            az += gz

    d = np.sqrt(ax * ax + ay * ay + az * az)
    d += 1e-12
    ax /= d
    ay /= d
    az /= d

    normal_mask = valids[0] | valids[1] | valids[2] | valids[3]
    invalid = ~normal_mask
    ax[invalid] = 0
    ay[invalid] = 0
    az[invalid] = 0
    return ax, ay, az, normal_mask


def _normals_edge_planes(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    tol: float,
    *,
    kernel_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """(H, W) pixels whose normal turns more than tol degrees within the window (as cos < cos(tol))."""
    d = np.sqrt(nx * nx + ny * ny + nz * nz)
    d += 1e-12
    nx = nx / d
    ny = ny / d
    nz = nz / d

    height, width = nx.shape
    pad = kernel_size // 2
    min_cos = np.ones((height, width), dtype=nx.dtype)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            c = slice(max(0, -dy), height - max(0, dy)), slice(max(0, -dx), width - max(0, dx))
            q = slice(max(0, dy), height - max(0, -dy)), slice(max(0, dx), width - max(0, -dx))
            cos = nx[c] * nx[q]
            cos += ny[c] * ny[q]
            cos += nz[c] * nz[q]
            if mask is not None:
                cos[~mask[q]] = 1.0  # masked neighbor contributes angle 0
            np.minimum(min_cos[c], cos, out=min_cos[c])
    return -_max_pool_2d(-min_cos, kernel_size) < np.cos(np.deg2rad(tol))


def edge_filter_mask(
    pts_cam: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    edge_rtol: float = 0.03,
    normals_tol: float = 5.0,
) -> np.ndarray:
    """Keep mask (S, H, W) dropping pixels on both a depth edge and a normal edge; pts_cam (S, H, W, 3), z = depth."""
    if valid_mask is None:
        valid_mask = pts_cam[..., 2] > 0
    keep = np.ones(pts_cam.shape[:3], dtype=bool)

    def one_frame(s):
        m = valid_mask[s]
        if not m.any():
            keep[s] = False
            return
        nx, ny, nz, normals_mask = _points_to_normal_planes(pts_cam[s], m)
        normal_edges = _normals_edge_planes(nx, ny, nz, tol=normals_tol, mask=normals_mask)
        depth_edges = depth_edge(pts_cam[s, ..., 2], rtol=edge_rtol, mask=m)
        keep[s] = ~(depth_edges & normal_edges)

    # frames are independent and numpy releases the GIL in these ops
    with ThreadPoolExecutor(max_workers=min(pts_cam.shape[0], os.cpu_count() or 1)) as pool:
        list(pool.map(one_frame, range(pts_cam.shape[0])))
    return keep


def baseline_filter_mask(pts_cam: np.ndarray, poses: np.ndarray, *, dist_alpha: float = 5.0) -> np.ndarray:
    """Keep mask (S, H, W) dropping pixels farther from their camera than dist_alpha x the first-last baseline."""
    max_dist = np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])
    if max_dist < 1e-6:
        return np.ones(pts_cam.shape[:3], dtype=bool)
    return np.linalg.norm(pts_cam, axis=-1) <= dist_alpha * max_dist


def apply_keep_mask(points: np.ndarray, confs: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero points and confidences at dropped pixels; shapes and grid order are preserved."""
    points_out = np.where(keep[..., None], points, 0.0).astype(points.dtype, copy=False)
    confs_out = np.where(keep, confs, 0.0).astype(confs.dtype, copy=False)
    return points_out, confs_out
