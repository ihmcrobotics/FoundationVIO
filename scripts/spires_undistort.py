"""Undistort an Oxford Spires sequence's front fisheye onto a centred f=700 pinhole (raw/cam0 -> raw/cam0_pinhole_f700).

Also writes sensor_pinhole_f700.yaml: the sequence's sensor.yaml with the pinhole intrinsics, extrinsics unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from foundation_vio.io.load import load_yaml

PINHOLE = [700.0, 700.0, 720.0, 540.0]  # fu, fv, pu, pv at the native 1440x1080


def _pinhole(fu, fv, pu, pv):
    return np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence_dir", type=Path, help="folder holding sensor.yaml and raw/cam0")
    args = parser.parse_args()

    sensor = load_yaml(args.sequence_dir / "sensor.yaml")
    cam = sensor["sensor"]["cameras"][0]  # cam_front, the camera the loader reads
    size = (int(cam["image_width"]), int(cam["image_height"]))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        _pinhole(*cam["intrinsics"]), np.array(cam["extra_params"]), np.eye(3), _pinhole(*PINHOLE), size, cv2.CV_16SC2
    )

    out = args.sequence_dir / "raw" / "cam0_pinhole_f700"
    out.mkdir(exist_ok=True)
    paths = [p for p in sorted((args.sequence_dir / "raw" / "cam0").glob("*.jpg")) if not (out / p.name).exists()]

    def undistort(p):
        img = cv2.remap(cv2.imread(str(p)), map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(str(out / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(undistort, paths))

    cam["intrinsics"] = PINHOLE
    cam["extra_params"] = [0.0, 0.0, 0.0, 0.0]
    (args.sequence_dir / "sensor_pinhole_f700.yaml").write_text(yaml.safe_dump(sensor, sort_keys=False))
    print(f"{len(paths)} images -> {out}")


if __name__ == "__main__":
    main()
