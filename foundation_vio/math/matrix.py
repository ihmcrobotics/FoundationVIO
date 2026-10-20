"""SE(3) helpers on 4x4 numpy matrices."""

import numpy as np
from scipy.spatial.transform import Rotation


def matrix_to_posquat(T):
    """[x, y, z, qx, qy, qz, qw]."""
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()])


def se3_inverse(T):
    R_inv = T[:3, :3].T
    inverse = np.eye(4)
    inverse[:3, :3] = R_inv
    inverse[:3, 3] = -R_inv @ T[:3, 3]
    return inverse


def transform_cloud_by_pose(points, T_new_old):
    """Apply T_new_old to a (..., 3) point array."""
    return (points.reshape(-1, 3) @ T_new_old[:3, :3].T + T_new_old[:3, 3]).reshape(points.shape)


def get_base_idx(batch_id, batch_size, batch_overlap):
    """(first image index, scale-node index) of a batch; the scale node is keyed by the batch's last image."""
    first = batch_id * (batch_size - batch_overlap)
    return first, first + batch_size - 1
