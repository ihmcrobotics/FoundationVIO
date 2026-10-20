"""ATE and RPE of a run's trajectory (SLAM, or odometry with loop_closure.enabled False) against TUM ground truth.

The run's poses are of the IMU; --gt-frame maps them to the ground truth's frame with the --calib extrinsics:
  imu   IMU in world (Aria: gt_poses_in_right_imu_frame_tum.txt); --calib is unused
  cam   camera, from a Kalibr camchain (T_cam_imu under --cam-key) or a Spires sensor yaml (ScaRF: poses_gt.txt)
  base  Spires base frame, from a sensor yaml (T_cam_imu, T_cam_lidar, T_base_lidar) (Spires: gt-tum.txt)
"""

import argparse
from pathlib import Path

import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

from foundation_vio.io.load import load_yaml


def t_xyz_q_xyzw(values):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    T[:3, 3] = values[:3]
    return T


def T_imu_gt(calib_path, gt_frame, cam_key):  # noqa: N802 - transform name
    """Pose of the ground-truth frame in the IMU frame."""
    if gt_frame == "imu":
        return np.eye(4)
    calib = load_yaml(calib_path)
    if "sensor" in calib:
        sensor = calib["sensor"]
        cam = sensor["cameras"][0]
        T_cam_imu = t_xyz_q_xyzw(cam["T_cam_imu_t_xyz_q_xyzw"])
    else:
        T_cam_imu = np.array(calib[cam_key]["T_cam_imu"], dtype=float).reshape(4, 4)
    T_imu_cam = np.linalg.inv(T_cam_imu)
    if gt_frame == "cam":
        return T_imu_cam
    if "sensor" not in calib:
        raise SystemExit("--gt-frame base needs a Spires sensor yaml (T_cam_lidar, T_base_lidar)")
    T_cam_lidar = t_xyz_q_xyzw(cam["T_cam_lidar_t_xyz_q_xyzw"])
    T_base_lidar = t_xyz_q_xyzw(sensor["T_base_lidar_t_xyz_q_xyzw"])
    return T_imu_cam @ T_cam_lidar @ np.linalg.inv(T_base_lidar)


def read_trajectory(path, T_imu_frame):
    """TUM IMU poses -> poses of the ground truth's frame."""
    rows = np.loadtxt(path, comments="#")
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    T[:, :3, 3] = rows[:, 1:4]
    return PoseTrajectory3D(poses_se3=list(T @ T_imu_frame), timestamps=rows[:, 0])


def ate(gt, est):
    """Translation APE RMSE [m] after SE(3) alignment, poses associated within 50 ms."""
    gt, est = sync.associate_trajectories(gt, est, max_diff=0.05)
    est.align(gt)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est))
    return ape.get_statistic(metrics.StatisticsType.rmse), est.num_poses


def rpe(gt, est):
    """Translational RPE RMSE [m/m] over 1 m path segments, poses associated within 50 ms."""
    gt, est = sync.associate_trajectories(gt, est, max_diff=0.05)
    rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0, delta_unit=metrics.Unit.meters)
    rpe.process_data((gt, est))
    return rpe.get_statistic(metrics.StatisticsType.rmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("gt", type=Path, help="ground truth, TUM format")
    parser.add_argument(
        "--calib", type=Path, help="Kalibr camchain or Spires sensor yaml (not needed for --gt-frame imu)"
    )
    parser.add_argument("--cam-key", default="cam0", help="camera block in a Kalibr camchain (default cam0)")
    parser.add_argument("--gt-frame", choices=["imu", "cam", "base"], default="cam")
    args = parser.parse_args()
    if args.gt_frame != "imu" and args.calib is None:
        parser.error("--calib is required unless --gt-frame imu")

    gt = file_interface.read_tum_trajectory_file(args.gt)
    T = T_imu_gt(args.calib, args.gt_frame, args.cam_key)
    est = read_trajectory(args.run_dir / "tum_poses.csv", T)
    lc = load_yaml(args.run_dir / "config.yaml")["loop_closure"].get("enabled", True)
    rmse, n = ate(gt, est)
    print(f"{'slam' if lc else 'odom'}: ATE {rmse:.3f} m, RPE {rpe(gt, est):.3f} m/m over {n} poses")


if __name__ == "__main__":
    main()
