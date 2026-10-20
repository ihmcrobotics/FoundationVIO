"""The factor graph of one batch: IMU, scale-pose and scale-scale factors and the first nodes' priors."""

from functools import partial

import gtsam
import numpy as np
import pandas as pd
from gtsam import CombinedImuFactor, CustomFactor, Pose3, PreintegrationCombinedParams, noiseModel
from gtsam.symbol_shorthand import B, S, V, X

from foundation_vio.graph.scale_pose_factor import error_func

HUBER_K = 1.345  # 95 %-efficiency Huber constant, for loop-closure factors


class Graph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.p = PreintegrationCombinedParams(np.array([0.0, 0.0, -9.81]))  # z-up world

    def set_imu_params(self, acc_noise, acc_rw, gyro_noise, gyro_rw, scale=1.0):
        """IMU noise densities and random walks; scale multiplies every covariance."""
        self.p.setAccelerometerCovariance(acc_noise**2 * np.eye(3) * scale)
        self.p.setGyroscopeCovariance(gyro_noise**2 * np.eye(3) * scale)
        self.p.setIntegrationCovariance(1e-8 * np.eye(3) * scale)
        self.p.setBiasAccCovariance(acc_rw**2 * np.eye(3) * scale)
        self.p.setBiasOmegaCovariance(gyro_rw**2 * np.eye(3) * scale)

    def preintegrate(self, seg, bias):
        """Preintegrate IMU rows (time [ns], acc_*, omega_*); row k covers time[k-1]..time[k]."""
        pim = gtsam.PreintegratedCombinedMeasurements(self.p, bias)
        dt = np.diff(seg["time"].to_numpy()) * 1e-9
        acc, gyro = seg[["acc_x", "acc_y", "acc_z"]].to_numpy(), seg[["omega_x", "omega_y", "omega_z"]].to_numpy()
        for k in range(1, len(seg)):
            if dt[k - 1] > 0:
                pim.integrateMeasurement(acc[k], gyro[k], float(dt[k - 1]))
        return pim

    def add_imu_factor(self, pim, i, j):
        self.graph.add(CombinedImuFactor(X(i), V(i), X(j), V(j), B(i), B(j), pim))

    def add_pose_prior(self, i, T, sigmas):
        self.graph.add(gtsam.PriorFactorPose3(X(i), Pose3(T), noiseModel.Diagonal.Sigmas(sigmas)))

    def add_vel_prior(self, i, v, sigmas):
        self.graph.add(gtsam.PriorFactorVector(V(i), v, noiseModel.Diagonal.Sigmas(sigmas)))

    def add_bias_prior(self, i, bias, sigmas):
        self.graph.add(gtsam.PriorFactorConstantBias(B(i), bias, noiseModel.Diagonal.Sigmas(sigmas)))

    def add_scale_prior(self, i, log_scale, sigma):
        self.graph.add(gtsam.PriorFactorDouble(S(i), log_scale, noiseModel.Diagonal.Sigmas(np.array([sigma]))))

    def add_vel_factor(self, i, j, sigmas):
        """Constant velocity between V(i) and V(j)."""
        self.graph.add(gtsam.BetweenFactorVector(V(i), V(j), np.zeros(3), noiseModel.Diagonal.Sigmas(sigmas)))

    def add_scale_pose_factor(self, i, j, s, T_ij, *, sigma_rot, sigma_trans, target_graph=None):
        """Model pose T_ij (model units) between X(i) and X(j) scaled by S(s); Huber when for target_graph (LC)."""
        noise = noiseModel.Gaussian.Covariance(np.diag([sigma_rot**2] * 3 + [sigma_trans**2] * 3))
        if target_graph is not None:
            noise = noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(HUBER_K), noise)
        graph = self.graph if target_graph is None else target_graph
        graph.add(CustomFactor(noise, [X(i), X(j), S(s)], partial(error_func, T_ij)))

    def add_scale_factor(self, i, j, log_ratio, sigma):
        """Factor log S(j) - log S(i) = log_ratio."""
        self.graph.add(gtsam.BetweenFactorDouble(S(i), S(j), log_ratio, noiseModel.Diagonal.Sigmas(np.array([sigma]))))

    @staticmethod
    def poses(values):
        """(index, 4x4 pose) of every X node, by index."""
        keys = [k for k in values.keys() if gtsam.Symbol(k).chr() == ord("x")]  # noqa: SIM118 - gtsam.Values
        return sorted((gtsam.Symbol(k).index(), values.atPose3(k).matrix()) for k in keys)

    @staticmethod
    def factor_errors(graph, values):
        rows = [(i, type(graph.at(i)).__name__, graph.at(i).error(values)) for i in range(graph.size())]
        return pd.DataFrame(rows, columns=["factor_num", "factor_type", "error"])
