"""Undistort an extracted ScaRF sequence with the dataset calibration (camera_rgb -> camera_rgb_undistorted)."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from foundation_vio.io.load import load_yaml


def _pinhole(fu, fv, pu, pv):
    return np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]])


def undistort_maps(config_dir):
    cam = load_yaml(config_dir / "kalibr_imucam_chain.yaml")["cam0"]
    rect = load_yaml(config_dir / "rectified_intrinsic.yaml")
    D = np.array(cam["distortion_coeffs"], dtype=np.float64)
    size = tuple(rect["pinhole_resolution"])  # (width, height)
    return cv2.fisheye.initUndistortRectifyMap(
        _pinhole(*cam["intrinsics"]), D, np.eye(3), _pinhole(*rect["pinhole_intrinsics"]), size, cv2.CV_16SC2
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="folder holding config/ and <sequence>/")
    parser.add_argument("sequence", help="r01 .. r05")
    args = parser.parse_args()

    images = args.dataset_dir / args.sequence / f"{args.sequence}_bag_sequence" / "images"
    out = images / "camera_rgb_undistorted"
    out.mkdir(exist_ok=True)
    map1, map2 = undistort_maps(args.dataset_dir / "config")
    paths = sorted((images / "camera_rgb").glob("*.jpg"))
    for p in paths:
        cv2.imwrite(
            str(out / p.name),
            cv2.remap(cv2.imread(str(p)), map1, map2, cv2.INTER_LINEAR),
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
    print(f"{len(paths)} images -> {out}")


if __name__ == "__main__":
    main()
