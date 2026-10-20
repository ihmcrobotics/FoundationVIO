"""ROS 2 bag -> <bag>_sequence/{images/camera_rgb, imu/imu_right/data.csv} (vendored from ori-drs/aria_ros)."""

import argparse
import csv
from pathlib import Path

import cv2
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag_dir", type=Path, help="ROS 2 bag folder (metadata.yaml + .mcap/.db3)")
    parser.add_argument("--image-topic", required=True)
    parser.add_argument("--imu-topic", required=True)
    parser.add_argument("--fps", type=float, default=20, help="maximum image rate kept (default 20)")
    args = parser.parse_args()

    out_dir = args.bag_dir.parent / f"{args.bag_dir.name}_sequence"
    images_dir = out_dir / "images" / "camera_rgb"
    imu_dir = out_dir / "imu" / "imu_right"
    images_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    min_gap_ns = 1e9 / args.fps

    count, last_capture_ns = 0, -min_gap_ns
    with (imu_dir / "data.csv").open("w", newline="") as f, AnyReader([args.bag_dir]) as reader:
        print("Topics:\n" + "\n".join(f"  {c.topic} - {c.msgtype}" for c in reader.connections))
        imu_csv = csv.writer(f)
        imu_csv.writerow(["#timestamp [ns]", "omega_x", "omega_y", "omega_z", "acc_x", "acc_y", "acc_z"])
        connections = [c for c in reader.connections if c.topic in (args.image_topic, args.imu_topic)]
        for connection, bag_ns, rawdata in reader.messages(connections=connections):
            if connection.topic == args.imu_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                w, a = msg.angular_velocity, msg.linear_acceleration
                imu_csv.writerow([stamp_ns(msg), w.x, w.y, w.z, a.x, a.y, a.z])
            elif bag_ns - last_capture_ns >= min_gap_ns:
                msg = reader.deserialize(rawdata, connection.msgtype)
                rgb = message_to_cvimage(msg, "rgb8")
                t = stamp_ns(msg)
                cv2.imwrite(
                    str(images_dir / f"image_{t // 1_000_000_000:010d}_{t % 1_000_000_000:09d}.jpg"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 100],
                )
                count, last_capture_ns = count + 1, bag_ns

    if count == 0:
        raise SystemExit(f"No images on {args.image_topic}; pick one of the topics listed above.")
    print(f"{count} images -> {images_dir}\nIMU -> {imu_dir / 'data.csv'}")


if __name__ == "__main__":
    main()
