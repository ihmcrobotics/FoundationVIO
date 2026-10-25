"""VI init: gravity for X(0) from IMU preintegration against one window of model poses (VINS' alignment)."""

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from foundation_vio.math.irls import estimate_scale

log = logging.getLogger(__name__)

MIN_EXCITATION = 0.25  # std of |accelerometer| over the window, m/s^2 (VINS' threshold)
GRAVITY_CHECK_DEG = 10.0
SLIDE_S = 1.0
MAX_ATTEMPTS = 60
GYRO_BIAS_ITERS = 3
G = 9.81


@dataclass
class VIInit:
    dR: np.ndarray  # noqa: N815 - correction for X(0)'s rotation: R_X0 = dR
    gravity_c0: np.ndarray  # gravity in the body frame of image 0
    gyro_bias: np.ndarray
    scale: float  # metric / model units (reported only)
    attempt: int


def _interp_rows(t_query, times, values):
    return np.array([np.interp(t_query, times, values[:, i]) for i in range(values.shape[1])])


def preintegrate(times_ns, acc, gyr, t_a_ns, t_b_ns, gyro_bias):
    """Midpoint preintegration over [t_a, t_b], bias removed: (dR, dv, dp, dt) in body_a, gravity kept."""
    inside = (times_ns > t_a_ns) & (times_ns < t_b_ns)
    t = np.concatenate(([t_a_ns], times_ns[inside], [t_b_ns])).astype(np.float64) * 1e-9
    a = np.vstack((_interp_rows(t_a_ns, times_ns, acc), acc[inside], _interp_rows(t_b_ns, times_ns, acc)))
    w = np.vstack((_interp_rows(t_a_ns, times_ns, gyr), gyr[inside], _interp_rows(t_b_ns, times_ns, gyr))) - gyro_bias
    R, v, p = np.eye(3), np.zeros(3), np.zeros(3)
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            continue
        R_next = R @ Rotation.from_rotvec(0.5 * (w[i] + w[i + 1]) * dt).as_matrix()
        a_mid = 0.5 * (R @ a[i] + R_next @ a[i + 1])
        p = p + v * dt + 0.5 * a_mid * dt * dt
        v = v + a_mid * dt
        R = R_next
    return R, v, p, float(t[-1] - t[0])


def solve_gyro_bias(preint_fn, R_rel_model, dts, iters=GYRO_BIAS_ITERS):
    """Gyro bias aligning preintegrated rotations with the model's (theta_k = -dt_k * bias, least squares)."""
    bg = np.zeros(3)
    for _ in range(iters):
        num, den = np.zeros(3), 0.0
        for k, R_rel in enumerate(R_rel_model):
            theta = Rotation.from_matrix(preint_fn(k, bg)[0].T @ R_rel).as_rotvec()
            num += -theta * dts[k]
            den += dts[k] ** 2
        bg = bg + num / den
    return bg


def _tangent_basis(g_dir):
    a = np.array([1.0, 0.0, 0.0]) if abs(g_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(g_dir, a)
    b1 /= np.linalg.norm(b1)
    return np.stack((b1, np.cross(g_dir, b1)), axis=1)


def linear_alignment(R, p_model, preints, g_mag=G):
    """VINS' LinearAlignment with body-frame model poses, then RefineGravity.

    Unknowns: velocities v_k in the c0 frame, gravity g in c0 (specific force at rest, |g| = g_mag, up) and the
    metric/model scale s. Rows per pair k -> k+1:
        -dt v_k + dt^2/2 g + s (p_{k+1} - p_k) = R_k dp_k
        -v_k + v_{k+1} + dt g = R_k dv_k
    Returns v (N,3), g (3,), s.
    """
    n = len(R)
    A = np.zeros((6 * (n - 1), 3 * n + 4))
    b = np.zeros(A.shape[0])
    gc, sc = slice(3 * n, 3 * n + 3), 3 * n + 3
    for k, (_, dv, dp, dt) in enumerate(preints):
        r = 6 * k
        A[r : r + 3, 3 * k : 3 * k + 3] = -dt * np.eye(3)
        A[r : r + 3, gc] = 0.5 * dt * dt * np.eye(3)
        A[r : r + 3, sc] = p_model[k + 1] - p_model[k]
        b[r : r + 3] = R[k] @ dp
        A[r + 3 : r + 6, 3 * k : 3 * k + 3] = -np.eye(3)
        A[r + 3 : r + 6, 3 * k + 3 : 3 * k + 6] = np.eye(3)
        A[r + 3 : r + 6, gc] = dt * np.eye(3)
        b[r + 3 : r + 6] = R[k] @ dv
    g = np.linalg.lstsq(A, b, rcond=None)[0][gc]
    for _ in range(4):  # RefineGravity: g = g_mag * g_dir + B w, re-solve for (v, w, s)
        g_dir = g / np.linalg.norm(g)
        B = _tangent_basis(g_dir)
        A2 = np.concatenate((A[:, : 3 * n], A[:, gc] @ B, A[:, [sc]]), axis=1)
        x = np.linalg.lstsq(A2, b - A[:, gc] @ (g_mag * g_dir), rcond=None)[0]
        g = g_mag * g_dir + B @ x[3 * n : 3 * n + 2]
    return x[: 3 * n].reshape(n, 3), g, float(x[-1])


def angle_deg(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_correction(gravity_body0, g_mag=G):
    """Roll/pitch dR making gravity point up; yaw = atan2(R[0,1], R[0,0]) as in the original R2ypr port."""
    v0, v1 = gravity_body0 / np.linalg.norm(gravity_body0), np.array([0.0, 0.0, 1.0])
    s = math.sqrt(max((1 + np.dot(v1, v0)) * 2, 1e-12))
    dR = Rotation.from_quat([*(np.cross(v0, v1) / s), s * 0.5]).as_matrix()
    yaw = math.atan2(dR[0, 1], dR[0, 0])
    c, sn = math.cos(-yaw), math.sin(-yaw)
    return np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]) @ dR


def model_window(loader, infer, first, last, *, batch_size, batch_overlap):
    """Model body poses for images first..last, chunked and scale-chained by IRLS on the overlap frame."""

    def own_frame(points, T_local_body):  # re-express one frame's points in its own body frame
        T = np.linalg.inv(T_local_body)
        return points @ T[:3, :3].T + T[:3, 3]

    step = batch_size - batch_overlap
    poses, prev, base = [], None, first
    while len(poses) < last - first + 1:
        paths = loader.get_batch_from_images(loader.image_path_list, base, batch_size)
        if len(paths) < batch_size:
            break
        points, bodies, confs = infer(paths)
        if prev is None:
            T_chain, new = np.eye(4), bodies
        else:
            p_points, p_bodies, p_confs, p_chain = prev
            s = estimate_scale(
                own_frame(p_points[step], p_bodies[step]), p_confs[step], own_frame(points[0], bodies[0]), confs[0]
            )
            bodies = [T.copy() for T in bodies]
            for T in bodies:
                T[:3, 3] *= s
            points = points * s
            T_chain = p_chain @ p_bodies[step] @ np.linalg.inv(bodies[0])  # chunk frame 0 == previous frame `step`
            new = bodies[batch_overlap:]
        poses.extend(T_chain @ T for T in new)
        prev = (points, bodies, confs, T_chain)
        base += step
    return poses[: last - first + 1]  # chunks can overshoot `last`


def run(loader, imu_df, timeshift_ns, window_fn, *, window_frames, g_mag=G):
    """First window that is IMU-excited and passes the gravity check: a VIInit, or None."""
    stamps_ns = np.array([loader.get_timestamp_ns(p) for p in loader.image_path_list], dtype=np.int64) + int(
        timeshift_ns
    )
    times_ns = imu_df["time"].to_numpy().astype(np.int64)
    acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float64)
    gyr = imu_df[["omega_x", "omega_y", "omega_z"]].to_numpy(dtype=np.float64)
    slide = max(1, round(SLIDE_S / (float(np.median(np.diff(stamps_ns))) * 1e-9)))

    rejected = {"low_excitation": 0, "short_model_window": 0, "gravity_check": 0}
    for attempt, first in enumerate(range(0, len(stamps_ns) - window_frames + 1, slide)[:MAX_ATTEMPTS]):
        last = first + window_frames - 1
        inside = (times_ns >= stamps_ns[first]) & (times_ns <= stamps_ns[last])
        excitation = float(np.std(np.linalg.norm(acc[inside], axis=1))) if inside.sum() > 2 else 0.0
        if excitation < MIN_EXCITATION:
            rejected["low_excitation"] += 1
            log.warning("VI-init window %d rejected: excitation %.4f < %.4f m/s^2 (%d IMU samples)",
                        first, excitation, MIN_EXCITATION, int(inside.sum()))
            continue

        T = window_fn(first, last)
        if len(T) < window_frames:
            rejected["short_model_window"] += 1
            log.warning("VI-init: model returned %d/%d required poses", len(T), window_frames)
            break
        T_rel = [np.linalg.inv(T[0]) @ Tk for Tk in T]  # c0 = body frame of the window's first frame
        R = [Tk[:3, :3] for Tk in T_rel]
        p_model = np.array([Tk[:3, 3] for Tk in T_rel])
        dts = [(stamps_ns[first + k + 1] - stamps_ns[first + k]) * 1e-9 for k in range(window_frames - 1)]

        def preint(k, bg, first=first):
            return preintegrate(times_ns, acc, gyr, stamps_ns[first + k], stamps_ns[first + k + 1], bg)

        bg = solve_gyro_bias(preint, [R[k].T @ R[k + 1] for k in range(window_frames - 1)], dts)
        preints = [preint(k, bg) for k in range(window_frames - 1)]
        _, g, s = linear_alignment(R, p_model, preints, g_mag)

        # the mean specific force over the window points along gravity unless the body accelerates throughout
        mean_force = sum(R[k] @ dv for k, (_, dv, _, _) in enumerate(preints)) / sum(dts)
        if angle_deg(mean_force, g) > GRAVITY_CHECK_DEG:
            rejected["gravity_check"] += 1
            log.warning("VI-init window %d rejected: gravity angle %.2f > %.2f degrees",
                        first, angle_deg(mean_force, g), GRAVITY_CHECK_DEG)
            continue

        # carry gravity from the window's first frame back to image 0 (bias-corrected gyro)
        R_0_first = preintegrate(times_ns, acc, gyr, stamps_ns[0], stamps_ns[first], bg)[0] if first else np.eye(3)
        gravity_body0 = R_0_first @ g
        log.info(
            f"VI-init: attempt {attempt} (image {first}): excitation {excitation:.2f}, scale {s:.3f}, "
            f"gyro bias {np.round(np.degrees(bg), 2).tolist()} deg/s"
        )
        return VIInit(rotation_correction(gravity_body0, g_mag), gravity_body0, bg, s, attempt)
    log.warning("VI-init: no window passed; X(0) keeps the identity rotation. Images=%d, required=%d, rejections=%s",
                len(stamps_ns), window_frames, rejected)
    return None
