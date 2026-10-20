"""The scale-pose factor's analytic Jacobians match numerical ones."""

import numpy as np
from gtsam import CustomFactor, Pose3, Rot3, Values, noiseModel
from gtsam.utils.numerical_derivative import numericalDerivative21, numericalDerivative22

from foundation_vio.graph.scale_pose_factor import error_func


def test_jacobians():
    factor = CustomFactor(noiseModel.Isotropic.Sigma(6, 0.1), [0, 1, 2], error_func)
    T_i = Pose3(Rot3.RzRyRx(0.1, 0.2, 0.3), np.array([1.0, 2.0, 3.0]))
    T_j = Pose3(Rot3.RzRyRx(0.4, 0.5, 0.6), np.array([4.0, 5.0, 6.0]))
    log_s = np.log(1.5)
    measurement = T_i.between(T_j).matrix()
    measurement[:3, 3] /= 1.5

    def err(Ti, Tj, l=log_s):
        v = Values()
        v.insert(0, Ti)
        v.insert(1, Tj)
        v.insert(2, l)
        return error_func(measurement, factor, v)

    v = Values()
    v.insert(0, T_i)
    v.insert(1, T_j)
    v.insert(2, log_s)
    H = [np.empty((6, 6)), np.empty((6, 6)), np.empty((6, 1))]
    error_func(measurement, factor, v, H)

    h = 1e-6
    numerical_l = ((err(T_i, T_j, log_s + h) - err(T_i, T_j, log_s - h)) / (2 * h)).reshape(6, 1)
    np.testing.assert_allclose(H[0], numericalDerivative21(err, T_i, T_j), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(H[1], numericalDerivative22(err, T_i, T_j), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(H[2], numerical_l, rtol=1e-5, atol=1e-7)
