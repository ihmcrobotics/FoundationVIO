"""Run artifacts: TUM trajectories and per-image clouds."""

import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import open3d as o3d
from tqdm import tqdm

from foundation_vio.mapping.fusion import fusion_keep
from foundation_vio.math.matrix import matrix_to_posquat, transform_cloud_by_pose

_o3d_lock = threading.Lock()  # Open3D is not documented as thread-safe


def write_tum(path, stamps_ns, poses):
    """One 'timestamp tx ty tz qx qy qz qw' line per pose (timestamp in seconds)."""
    with path.open("w") as f:
        f.write("#timestamp tx ty tz qx qy qz qw\n")
        for t, T in zip(stamps_ns, poses, strict=True):
            f.write(f"{t // 1_000_000_000}.{t % 1_000_000_000:09d} " + " ".join(map(str, matrix_to_posquat(T))) + "\n")


def quat_to_matrix(q):
    """(N, 4) quaternions x, y, z, w -> (N, 3, 3) rotation matrices."""
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )


def load_tum(path):
    """(times [s], (N, 4, 4) poses) from a TUM file 'timestamp tx ty tz qx qy qz qw'."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = quat_to_matrix(rows[:, 4:8])
    T[:, :3, 3] = rows[:, 1:4]
    return rows[:, 0], T


def write_ply(path, points, colors):
    """Colors 0-255 (or 0-1). Arrays go to Open3D float64 and C-contiguous: a strided buffer segfaults it."""
    colors = np.asarray(colors, dtype=np.float64)
    if colors.max() > 1.0:
        colors = colors / 255.0
    with _o3d_lock:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64, order="C")))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, order="C"))
        o3d.io.write_point_cloud(str(path), cloud)


def world_points(submap, *, max_range, conf_percentile):
    """World points, confs, colors; conf 0 marks rejected points and the export cuts (range, low-confidence %)."""
    points, confs, colors = submap.get_points(frame_num=None, ref_num=0)
    points = transform_cloud_by_pose(points, submap.T_world_body0).astype(np.float32, copy=False)
    confs = np.array(confs).reshape(-1)
    n = submap.H * submap.W
    if max_range > 0:
        for i, T in enumerate(submap.T_local_bodies):
            sl = slice(i * n, (i + 1) * n)
            confs[sl][np.linalg.norm(points[sl] - (submap.T_world_body0 @ T)[:3, 3], axis=1) > max_range] = 0.0
    if conf_percentile > 0:
        positive = confs[confs > 0]
        if positive.size:
            confs[confs < np.percentile(positive, conf_percentile)] = 0.0
    return points, confs, colors


def export_submaps(global_map, out_dir, timestamps_ns, stride, *, max_range, conf_percentile):
    """Write each image's points, one per fused surface, through a thread pool (4 queued writes per worker)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def drain(limit):
            while len(pending) > limit:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    pending.discard(f)

        for submap in tqdm(global_map.batches(), desc="Saving submaps", unit="submap"):
            spilled = submap.points is None
            if spilled:
                submap.reload()
            points, confs, colors = world_points(submap, max_range=max_range, conf_percentile=conf_percentile)
            keep = confs > 0 if submap.fusion_roots is None else fusion_keep(confs, submap.fusion_roots)
            n = submap.H * submap.W
            for i in range(submap.S):
                img_idx = submap.batch_id * stride + i
                if img_idx >= len(timestamps_ns):
                    break
                frame = slice(i * n, (i + 1) * n)
                k = keep[frame]
                if k.any():
                    t = timestamps_ns[img_idx]
                    path = out_dir / f"cloud_{t // 1_000_000_000:010d}_{t % 1_000_000_000:09d}.ply"
                    pending.add(pool.submit(write_ply, path, points[frame][k], colors[frame][k]))
            if spilled:
                submap.release_points()
            drain(4 * workers)
        drain(0)
