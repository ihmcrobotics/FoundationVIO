"""Synthetic VI init: known gravity, gyro bias and scale come back."""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from foundation_vio.math import vi_init

G = 9.81
RNG = np.random.default_rng(0)


def make_sequence(
    duration_s=3.0, imu_hz=1000, frame_hz=6, gyro_bias=(0.01, -0.02, 0.015), scale=0.8, *, accelerate=True
):
    """Body moving on a smooth path with rotation; accelerometer = R^T (a + g_up), gyro = omega + bias."""
    t = np.arange(0, duration_s, 1.0 / imu_hz)
    # trajectory in the world (z up): sinusoidal position, slowly varying yaw/pitch
    amp = 0.4 if accelerate else 0.0
    bounce = 0.05 if accelerate else 0.0  # walking-like vertical oscillation, ~1.8 m/s^2 peak
    p = np.stack((1.0 * t + amp * np.sin(2 * t), amp * np.cos(1.5 * t), bounce * np.sin(6 * t)), axis=1)
    v = np.gradient(p, t, axis=0)
    a = np.gradient(v, t, axis=0)
    ypr = np.stack((0.3 * np.sin(0.7 * t), 0.15 * np.sin(1.1 * t), 0.1 * np.cos(0.9 * t)), axis=1)
    R = Rotation.from_euler("zyx", ypr).as_matrix()
    g_up = np.array([0.0, 0.0, G])
    acc_body = np.einsum("nji,nj->ni", R, a + g_up)  # R^T (a + g)
    # body rates from consecutive rotations
    dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
    omega = Rotation.from_matrix(dR).as_rotvec() * imu_hz
    omega = np.vstack((omega, omega[-1:]))
    gyro_body = omega + np.asarray(gyro_bias)
    imu = pd.DataFrame(
        {
            "time": (t * 1e9).astype(np.int64),
            "acc_x": acc_body[:, 0],
            "acc_y": acc_body[:, 1],
            "acc_z": acc_body[:, 2],
            "omega_x": gyro_body[:, 0],
            "omega_y": gyro_body[:, 1],
            "omega_z": gyro_body[:, 2],
        }
    )
    frame_idx = np.arange(0, len(t), imu_hz // frame_hz)
    poses = np.tile(np.eye(4), (len(frame_idx), 1, 1))
    poses[:, :3, :3] = R[frame_idx]
    poses[:, :3, 3] = p[frame_idx] / scale  # the model's poses are the metric ones divided by the scale
    return imu, t[frame_idx], poses, R[frame_idx[0]].T @ g_up


class FakeLoader:
    def __init__(self, stamps_s):
        self.image_path_list = [f"frame_{i:04d}.jpg" for i in range(len(stamps_s))]
        self._stamps = {path: round(s * 1e9) for path, s in zip(self.image_path_list, stamps_s, strict=True)}

    def get_timestamp_ns(self, path):
        return self._stamps[path]


def run(imu, stamps, poses, window_frames=13):
    return vi_init.run(FakeLoader(stamps), imu, 0, lambda a, b: list(poses[a : b + 1]), window_frames=window_frames)


def test_recovers_gravity_bias_and_scale():
    imu, stamps, poses, g_c0 = make_sequence()
    res = run(imu, stamps, poses)
    assert res is not None
    assert vi_init.angle_deg(res.gravity_c0, g_c0) < 0.3
    assert np.allclose(res.gyro_bias, (0.01, -0.02, 0.015), atol=5e-4)
    assert abs(res.scale - 0.8) < 0.03
    assert res.attempt == 0


def test_constant_velocity_still_gives_gravity(monkeypatch):
    """No acceleration: the scale is unobservable but gravity is right."""
    monkeypatch.setattr(vi_init, "MIN_EXCITATION", 0.0)
    imu, stamps, poses, g_c0 = make_sequence(accelerate=False, scale=1.0)
    res = run(imu, stamps, poses)
    assert res is not None
    assert vi_init.angle_deg(res.gravity_c0, g_c0) < 0.5


def test_rotation_correction_levels_gravity():
    g = np.array([0.5, -0.3, 9.7])
    assert vi_init.angle_deg(vi_init.rotation_correction(g) @ g, [0, 0, 1]) < 1e-6


def test_preintegration_at_rest_returns_gravity():
    imu, _, _, _ = make_sequence(accelerate=False, scale=1.0)
    times = imu["time"].to_numpy()
    acc = imu[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyr = imu[["omega_x", "omega_y", "omega_z"]].to_numpy()
    _, dv, _, dt = vi_init.preintegrate(times, acc, gyr, times[0], times[0] + int(0.2e9), np.zeros(3))
    assert abs(dt - 0.2) < 1e-6
    assert abs(np.linalg.norm(dv) / dt - vi_init.G) < 0.2  # constant velocity: specific force is gravity
