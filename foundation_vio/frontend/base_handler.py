"""Feed-forward model interface: one call per batch of images, geometry in one convention."""

from typing import NamedTuple

import numpy as np


class Prediction(NamedTuple):
    pts_cam: np.ndarray  # (S, H, W, 3) points in each camera frame (z = depth)
    points: np.ndarray  # (S, H, W, 3) points in the model's local frame
    poses: np.ndarray  # (S, 4, 4) camera-to-local
    confs: np.ndarray  # (S, H, W)
    masks: np.ndarray  # (S, H, W) valid depth
    colors: np.ndarray  # (S, H, W, 3) uint8, at the output resolution
    K: np.ndarray  # (S, 3, 3) intrinsics at the output resolution (the calibrated K, rescaled)


def scale_K(K, orig_size, out_hw):
    """Calibrated K of an orig_size (W, H) image, rescaled to the model's (H, W) output."""
    K = np.asarray(K, dtype=np.float32).copy()
    K[0, :] *= out_hw[1] / orig_size[0]
    K[1, :] *= out_hw[0] / orig_size[1]
    return K


def unproject(depth, K, c2w):
    """Depth (S, H, W) with intrinsics (S, 3, 3) and camera-to-local poses (S, 4, 4) -> (pts_cam, pts_local)."""
    S, H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts_cam, pts_local = [], []
    for i in range(S):
        d = depth[i]
        pts_c = np.stack([(uu - K[i, 0, 2]) / K[i, 0, 0] * d, (vv - K[i, 1, 2]) / K[i, 1, 1] * d, d], axis=-1)
        pts_cam.append(pts_c)
        pts_local.append(((c2w[i, :3, :3] @ pts_c.reshape(-1, 3).T).T + c2w[i, :3, 3]).reshape(H, W, 3))
    return np.stack(pts_cam), np.stack(pts_local)


def w2c_to_c2w(E):
    """(S, 3, 4) or (S, 4, 4) world-to-camera -> (S, 4, 4) camera-to-world."""
    E = E.astype(np.float32)
    if E.shape[-2:] == (3, 4):
        E44 = np.tile(np.eye(4, dtype=np.float32)[None], (len(E), 1, 1))
        E44[:, :3, :] = E
        E = E44
    return np.linalg.inv(E)
