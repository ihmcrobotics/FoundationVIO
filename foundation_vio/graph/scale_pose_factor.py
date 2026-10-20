"""Scale-pose factor: a model relative pose (up to scale) between two pose nodes, scaled by a log-scale node."""

import numpy as np
from gtsam import CustomFactor, Rot3, Values


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def right_jacobian_SO3_inv(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * skew(phi) + (1 / 12) * skew(phi) @ skew(phi)
    axis, half = phi / theta, 0.5 * theta
    cot_half = 1.0 / np.tan(half)
    return (half * cot_half) * np.eye(3) + (1 - half * cot_half) * np.outer(axis, axis) + half * skew(axis)


def error_func(measurement: np.ndarray, this: CustomFactor, v: Values, H: list[np.ndarray] | None = None):
    """Residual [Log(Z_R^T R_i^T R_j), Z_t - exp(l) R_i^T (t_j - t_i)] for keys (X_i, X_j, S_l), with Jacobians."""
    key_i, key_j, key_l = this.keys()
    T_i, T_j = v.atPose3(key_i), v.atPose3(key_j)
    s = np.exp(v.atDouble(key_l))
    R_i, t_i = T_i.rotation().matrix(), T_i.translation()
    R_j, t_j = T_j.rotation().matrix(), T_j.translation()
    Z_R, Z_t = measurement[:3, :3], measurement[:3, 3]

    rR = Rot3.Logmap(Rot3(Z_R.T @ (R_i.T @ R_j)))
    u = R_i.T @ (t_j - t_i)  # relative translation in frame i
    if H is not None:
        Jr_inv = right_jacobian_SO3_inv(rR)
        H[0] = np.block([[-Jr_inv @ R_j.T @ R_i, np.zeros((3, 3))], [-s * skew(u), s * np.eye(3)]])
        H[1] = np.block([[Jr_inv, np.zeros((3, 3))], [np.zeros((3, 3)), -s * R_i.T @ R_j]])
        H[2] = np.vstack([np.zeros((3, 1)), -s * u.reshape(3, 1)])
    return np.hstack([rR, Z_t - s * u])
