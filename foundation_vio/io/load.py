"""Dataset inputs; dataset_type 'default' (Kalibr YAMLs: Aria, ScaRF) or 'oxford_spires' (sensor.yaml)."""

import pathlib
import re

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_yaml(path):
    """Parse a YAML file, tolerating the OpenCV/Kalibr '%YAML:1.0' header PyYAML rejects."""
    text = pathlib.Path(path).read_text()
    first_line, _, rest = text.partition("\n")
    if first_line.lstrip().startswith("%YAML:"):
        text = rest
    return yaml.safe_load(text)


def sequence_dir(config):
    return pathlib.Path(config["files"]["dataset_dir"]).expanduser() / config["sequence"]["sequence_name"]


def make_loader(config):
    """The loader for config's dataset type, with every input file resolved against the sequence directory."""
    files, seq = config["files"], sequence_dir(config)
    if not seq.is_dir():
        raise FileNotFoundError(
            f"{seq} not found: files.dataset_dir ({files['dataset_dir']}) must be the folder that contains "
            f"sequence.sequence_name ({config['sequence']['sequence_name']})"
        )
    loaders = {"default": DataLoader, "oxford_spires": OxfordSpiresDataLoader}
    dataset_type = config["sequence"].get("dataset_type", "default")
    if dataset_type not in loaders:
        raise ValueError(f"sequence.dataset_type must be one of {sorted(loaders)}, got {dataset_type!r}")
    return loaders[dataset_type](
        image_dir=seq / files["image_dir"],
        imu_file=seq / files["imu_file"],
        intrinsics_file=seq / files["intrinsics_file"],
        extrinsics_file=seq / files.get("extrinsics_file", files["intrinsics_file"]),
        imu_params_file=_imu_params_file(seq, files),
        camera_key=files.get("camera_key", "cam2"),
    )


def _imu_params_file(seq, files):
    """Under the sequence directory if present, otherwise cwd-relative (configs/imu_spires_x10.yaml)."""
    p = pathlib.Path(files["imu_params_file"]).expanduser()
    if p.is_absolute():
        return p
    at_seq = seq / p
    return at_seq if at_seq.is_file() else p


def _pinhole(fu, fv, pu, pv):
    return np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]])


def stamp_ns(path):
    """Last two integers in the stem: image_<s>_<ns>.jpg, cloud_<s>_<ns>.ply, <s>.<ns>.jpg."""
    numbers = re.findall(r"\d+", pathlib.Path(path).stem)
    if len(numbers) < 2:
        raise ValueError(f"Unexpected filename format: {pathlib.Path(path).name}")
    s, ns = numbers[-2], numbers[-1].ljust(9, "0")[:9]
    return int(s) * 1_000_000_000 + int(ns)


class DataLoader:
    def __init__(self, *, image_dir, imu_file, intrinsics_file, extrinsics_file, imu_params_file, camera_key="cam2"):
        self.image_dir = pathlib.Path(image_dir)
        self.imu_file = imu_file
        self.intrinsics_file = intrinsics_file
        self.extrinsics_file = extrinsics_file
        self.imu_params_file = imu_params_file
        self.camera_key = camera_key  # the camera's block in a Kalibr camchain (Aria cam2, ScaRF cam0)
        self.image_path_list = None
        self.T_BC = None
        self.T_CB = None

    # --- images ---

    get_timestamp_ns = staticmethod(stamp_ns)

    def fetch_image_from_dir(self, img_stride=1):
        """Every img_stride-th image of image_dir, in time order."""
        paths = sorted(
            (p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=self.get_timestamp_ns
        )
        if not paths:
            raise ValueError(f"No images found in {self.image_dir}")
        self.image_path_list = paths[::img_stride]
        return self.image_path_list

    def get_batch_from_images(self, image_path_list, start_idx, batch_size):
        return sorted(image_path_list[start_idx : start_idx + batch_size], key=self.get_timestamp_ns)

    # --- IMU ---

    def fetch_imu_data(self):
        """IMU samples as columns time [ns], omega_x/y/z, acc_x/y/z."""
        imu_df = pd.read_csv(self.imu_file)
        imu_df.columns = ["time", "omega_x", "omega_y", "omega_z", "acc_x", "acc_y", "acc_z"]
        return imu_df

    def fetch_imu_params(self):
        """(acc noise, acc random walk, gyro noise, gyro random walk)."""
        imu = load_yaml(self.imu_params_file)
        imu = imu.get("imu0", imu)  # Kalibr nests the block under imu0
        return (
            imu["accelerometer_noise_density"],
            imu["accelerometer_random_walk"],
            imu["gyroscope_noise_density"],
            imu["gyroscope_random_walk"],
        )

    # --- calibration ---

    def fetch_T_cam_imu(self):
        """T_CB (camera from IMU) from the camchain; also sets T_CB and T_BC."""
        self.T_CB = np.array(load_yaml(self.extrinsics_file)[self.camera_key]["T_cam_imu"]).reshape(4, 4)
        self.T_BC = np.linalg.inv(self.T_CB)
        return self.T_CB

    def fetch_timeshift_cam_imu(self):
        """Camera-to-IMU time offset in ns (image_time + offset = imu_time); 0 when not calibrated."""
        cam = load_yaml(self.extrinsics_file).get(self.camera_key, {})
        return int(round(cam.get("timeshift_cam_imu", 0.0) * 1e9))

    def fetch_T_cam_intrinsics(self):
        """K from a flat 'pinhole_intrinsics: [fu, fv, pu, pv]' or the camchain block's 'intrinsics'."""
        calib = load_yaml(self.intrinsics_file)
        line = np.array(
            calib["pinhole_intrinsics"] if "pinhole_intrinsics" in calib else calib[self.camera_key]["intrinsics"],
            dtype=np.float64,
        )
        return _pinhole(*line[:4])


class OxfordSpiresDataLoader(DataLoader):
    """Oxford Spires: calibration in sensor.yaml (camera 0), IMU CSV with secs/nsecs columns."""

    def fetch_imu_data(self):
        imu_df = pd.read_csv(self.imu_file)
        imu_df["time"] = (imu_df["secs"] * 1_000_000_000 + imu_df["nsecs"]).astype(np.int64)
        imu_df = imu_df[["time", "ang_vel_x", "ang_vel_y", "ang_vel_z", "acc_x", "acc_y", "acc_z"]]
        imu_df.columns = ["time", "omega_x", "omega_y", "omega_z", "acc_x", "acc_y", "acc_z"]
        return imu_df

    def fetch_T_cam_imu(self):
        camera = load_yaml(self.extrinsics_file)["sensor"]["cameras"][0]
        t_xyz_q_xyzw = np.array(camera["T_cam_imu_t_xyz_q_xyzw"], dtype=np.float64)
        self.T_CB = np.eye(4)
        self.T_CB[:3, :3] = Rotation.from_quat(t_xyz_q_xyzw[3:]).as_matrix()
        self.T_CB[:3, 3] = t_xyz_q_xyzw[:3]
        self.T_BC = np.linalg.inv(self.T_CB)
        return self.T_CB

    def fetch_timeshift_cam_imu(self):
        return 0  # sensor.yaml carries no cam-imu time offset

    def fetch_T_cam_intrinsics(self):
        line = np.array(load_yaml(self.intrinsics_file)["sensor"]["cameras"][0]["intrinsics"], dtype=np.float64)
        return _pinhole(*line[:4])
